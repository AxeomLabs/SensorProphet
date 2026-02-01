"""
Early Warning Prediction Module

Predicts potential anomalies 2-48 hours in advance by analyzing
sensor trends and forecasting future values.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.config import FORECAST_HORIZONS
from src.utils import setup_logging

logger = setup_logging("early_warning")

# Early warning configuration
EARLY_WARNING_HORIZONS = [2, 6, 12, 24, 48]  # Hours to predict ahead
EARLY_WARNING_CONFIDENCE_THRESHOLD = 0.7
EARLY_WARNING_MIN_SAMPLES = 20


class PredictionSeverity(Enum):
    """Severity levels for predictions."""
    NORMAL = "normal"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PredictedAnomaly:
    """Predicted anomaly with timing and confidence."""
    predicted_time_hours: float
    confidence: float
    severity: PredictionSeverity
    affected_metrics: List[str]
    current_values: Dict[str, float]
    predicted_values: Dict[str, float]
    thresholds: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "predicted_time_hours": self.predicted_time_hours,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "affected_metrics": self.affected_metrics,
            "current_values": self.current_values,
            "predicted_values": self.predicted_values,
            "thresholds": self.thresholds,
            "recommendations": self.recommendations,
        }


@dataclass
class EarlyWarningResult:
    """Result of early warning analysis."""
    timestamp: str
    has_warning: bool
    earliest_anomaly_hours: Optional[float]
    predictions: List[PredictedAnomaly]
    degradation_rate: Optional[float]
    remaining_useful_life_hours: Optional[float]
    overall_risk_score: float  # 0-100, higher = more risk
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "has_warning": self.has_warning,
            "earliest_anomaly_hours": self.earliest_anomaly_hours,
            "predictions": [p.to_dict() for p in self.predictions],
            "degradation_rate": self.degradation_rate,
            "remaining_useful_life_hours": self.remaining_useful_life_hours,
            "overall_risk_score": self.overall_risk_score,
        }


class EarlyWarningPredictor:
    """Predicts anomalies 2-48 hours in advance using trend analysis and forecasting."""
    
    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        confidence_threshold: float = EARLY_WARNING_CONFIDENCE_THRESHOLD,
        min_samples: int = EARLY_WARNING_MIN_SAMPLES,
    ):
        """
        Initialize early warning predictor.
        
        Args:
            horizons: Prediction horizons in hours (default: [2, 6, 12, 24, 48]).
            confidence_threshold: Minimum confidence to trigger warning.
            min_samples: Minimum samples needed for prediction.
        """
        self.horizons = horizons or EARLY_WARNING_HORIZONS
        self.confidence_threshold = confidence_threshold
        self.min_samples = min_samples
        
        # History buffer for trend analysis
        self._history: List[Dict] = []
        self._timestamps: List[datetime] = []
        
        # Thresholds for anomaly detection (will be calibrated)
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        
        self.is_calibrated = False
    
    def calibrate(
        self,
        baseline_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        anomaly_percentile: float = 95,
    ) -> "EarlyWarningPredictor":
        """
        Calibrate thresholds from baseline (healthy) data.
        
        Args:
            baseline_data: DataFrame with healthy equipment data.
            feature_columns: Columns to use for prediction.
            anomaly_percentile: Percentile to use as anomaly threshold.
            
        Returns:
            Self for chaining.
        """
        if feature_columns is None:
            feature_columns = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = feature_columns
        
        for col in feature_columns:
            values = baseline_data[col].dropna().values
            
            # Calculate baseline statistics
            self._baseline_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            
            # Calculate thresholds (values above these indicate degradation)
            self._thresholds[col] = {
                "warning": float(np.percentile(values, anomaly_percentile)),
                "critical": float(np.percentile(values, 99)),
            }
        
        self.is_calibrated = True
        logger.info(f"Calibrated early warning on {len(feature_columns)} features")
        return self
    
    def add_sample(
        self,
        sample: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Add a sample to the history buffer.
        
        Args:
            sample: Dictionary of feature values.
            timestamp: Sample timestamp.
        """
        timestamp = timestamp or datetime.now()
        self._history.append(sample)
        self._timestamps.append(timestamp)
        
        # Keep buffer at reasonable size (last 500 samples)
        if len(self._history) > 500:
            self._history.pop(0)
            self._timestamps.pop(0)
    
    def predict(
        self,
        current_sample: Optional[Dict[str, float]] = None,
    ) -> EarlyWarningResult:
        """
        Predict potential anomalies based on current trends.
        
        Args:
            current_sample: Current sensor readings (optional, uses latest from history).
            
        Returns:
            EarlyWarningResult with predictions.
        """
        timestamp = datetime.now().isoformat()
        
        # Add current sample to history if provided
        if current_sample:
            self.add_sample(current_sample)
        
        # Check if we have enough data
        if len(self._history) < self.min_samples:
            return EarlyWarningResult(
                timestamp=timestamp,
                has_warning=False,
                earliest_anomaly_hours=None,
                predictions=[],
                degradation_rate=None,
                remaining_useful_life_hours=None,
                overall_risk_score=0.0,
            )
        
        # Analyze each feature
        predictions = []
        degradation_rates = []
        ruls = []
        
        for col in self.feature_columns:
            if col not in self._history[-1]:
                continue
            
            # Get historical values for this feature
            values = [h.get(col, 0) for h in self._history if col in h]
            if len(values) < self.min_samples:
                continue
            
            values = np.array(values)
            
            # Analyze trend and make predictions
            prediction = self._predict_feature(col, values)
            if prediction:
                predictions.append(prediction)
            
            # Calculate degradation rate
            deg_rate = self._calculate_degradation_rate(values)
            if deg_rate is not None:
                degradation_rates.append(deg_rate)
            
            # Estimate RUL for this feature
            rul = self._estimate_rul(col, values)
            if rul is not None:
                ruls.append(rul)
        
        # Aggregate results
        has_warning = len(predictions) > 0
        earliest_hours = min([p.predicted_time_hours for p in predictions]) if predictions else None
        avg_degradation = float(np.mean(degradation_rates)) if degradation_rates else None
        min_rul = min(ruls) if ruls else None
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(predictions, avg_degradation, min_rul)
        
        return EarlyWarningResult(
            timestamp=timestamp,
            has_warning=has_warning,
            earliest_anomaly_hours=earliest_hours,
            predictions=predictions,
            degradation_rate=avg_degradation,
            remaining_useful_life_hours=min_rul,
            overall_risk_score=risk_score,
        )
    
    def _predict_feature(
        self,
        feature_name: str,
        values: np.ndarray,
    ) -> Optional[PredictedAnomaly]:
        """Predict anomaly for a single feature."""
        if feature_name not in self._thresholds:
            return None
        
        thresholds = self._thresholds[feature_name]
        baseline = self._baseline_stats[feature_name]
        current_value = float(values[-1])
        
        # Fit linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Check if degrading (for metrics where higher = worse)
        is_increasing = slope > 0
        # Determine if higher or lower is bad based on baseline
        higher_is_bad = current_value > baseline["mean"]
        
        if not ((higher_is_bad and is_increasing) or (not higher_is_bad and not is_increasing)):
            # Not trending toward anomaly
            return None
        
        # Project when threshold will be crossed
        for horizon in self.horizons:
            # Estimate samples per hour (assume ~1 sample per data interval)
            samples_per_hour = max(1, len(values) / 24)  # Rough estimate
            future_idx = len(values) + int(horizon * samples_per_hour)
            predicted_value = intercept + slope * future_idx
            
            # Check warning threshold
            threshold_to_check = thresholds["warning"]
            will_exceed = (predicted_value > threshold_to_check) if higher_is_bad else (predicted_value < -threshold_to_check)
            
            if will_exceed:
                # Calculate confidence based on trend strength and proximity
                trend_strength = abs(slope) / (baseline["std"] + 0.001)
                proximity = abs(current_value - threshold_to_check) / (abs(predicted_value - current_value) + 0.001)
                confidence = min(1.0, trend_strength * 0.5 + (1.0 / (proximity + 1)) * 0.5)
                
                if confidence >= self.confidence_threshold:
                    # Determine severity based on horizon
                    if horizon <= 6:
                        severity = PredictionSeverity.CRITICAL
                    elif horizon <= 24:
                        severity = PredictionSeverity.WARNING
                    else:
                        severity = PredictionSeverity.INFO
                    
                    # Generate recommendations
                    recommendations = self._generate_recommendations(
                        feature_name, horizon, severity
                    )
                    
                    return PredictedAnomaly(
                        predicted_time_hours=float(horizon),
                        confidence=float(confidence),
                        severity=severity,
                        affected_metrics=[feature_name],
                        current_values={feature_name: current_value},
                        predicted_values={feature_name: float(predicted_value)},
                        thresholds={feature_name: float(threshold_to_check)},
                        recommendations=recommendations,
                    )
        
        return None
    
    def _calculate_degradation_rate(self, values: np.ndarray) -> Optional[float]:
        """Calculate degradation rate (normalized slope)."""
        if len(values) < 5:
            return None
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Normalize by mean value
        mean_val = np.mean(values)
        if mean_val != 0:
            return float(slope / abs(mean_val) * 100)  # Percentage per sample
        return float(slope)
    
    def _estimate_rul(
        self,
        feature_name: str,
        values: np.ndarray,
    ) -> Optional[float]:
        """Estimate Remaining Useful Life in hours."""
        if feature_name not in self._thresholds:
            return None
        
        threshold = self._thresholds[feature_name]["critical"]
        current_value = values[-1]
        
        # Check if already past threshold
        if current_value >= threshold:
            return 0.0
        
        # Linear projection
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        if slope <= 0:
            return None  # Not degrading
        
        # Steps until threshold
        current_step = len(values) - 1
        threshold_step = (threshold - intercept) / slope
        steps_remaining = threshold_step - current_step
        
        if steps_remaining <= 0:
            return 0.0
        
        # Convert to hours (rough estimate)
        samples_per_hour = max(1, len(values) / 24)
        hours_remaining = steps_remaining / samples_per_hour
        
        return float(max(0, hours_remaining))
    
    def _calculate_risk_score(
        self,
        predictions: List[PredictedAnomaly],
        degradation_rate: Optional[float],
        rul: Optional[float],
    ) -> float:
        """Calculate overall risk score (0-100)."""
        if not predictions and not degradation_rate:
            return 0.0
        
        score = 0.0
        
        # Risk from predictions
        for pred in predictions:
            if pred.severity == PredictionSeverity.CRITICAL:
                score += 40 * pred.confidence
            elif pred.severity == PredictionSeverity.WARNING:
                score += 25 * pred.confidence
            else:
                score += 10 * pred.confidence
        
        # Risk from degradation rate
        if degradation_rate and degradation_rate > 0:
            score += min(20, degradation_rate * 10)
        
        # Risk from RUL
        if rul is not None:
            if rul < 6:
                score += 30
            elif rul < 24:
                score += 15
            elif rul < 48:
                score += 5
        
        return min(100.0, score)
    
    def _generate_recommendations(
        self,
        feature: str,
        hours: float,
        severity: PredictionSeverity,
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []
        
        if severity == PredictionSeverity.CRITICAL:
            recommendations.append(f"URGENT: Schedule immediate inspection within {hours:.0f} hours")
            recommendations.append("Prepare backup equipment if available")
            recommendations.append(f"Monitor {feature} closely for rapid degradation")
        elif severity == PredictionSeverity.WARNING:
            recommendations.append(f"Schedule maintenance within {hours:.0f} hours")
            recommendations.append(f"Investigate root cause of {feature} degradation")
        else:
            recommendations.append(f"Plan maintenance check within {hours:.0f} hours")
            recommendations.append(f"Add {feature} to watch list")
        
        return recommendations
    
    def get_history_summary(self) -> Dict:
        """Get summary of collected history."""
        if not self._history:
            return {"samples": 0}
        
        return {
            "samples": len(self._history),
            "oldest": self._timestamps[0].isoformat() if self._timestamps else None,
            "newest": self._timestamps[-1].isoformat() if self._timestamps else None,
            "features": list(self._history[-1].keys()) if self._history else [],
        }
    
    def reset(self) -> None:
        """Reset history buffer."""
        self._history.clear()
        self._timestamps.clear()


def create_early_warning_predictor(
    baseline_data: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
) -> EarlyWarningPredictor:
    """
    Factory function to create and optionally calibrate an EarlyWarningPredictor.
    
    Args:
        baseline_data: Optional baseline data for calibration.
        feature_columns: Optional list of feature columns.
        
    Returns:
        Configured EarlyWarningPredictor.
    """
    predictor = EarlyWarningPredictor()
    
    if baseline_data is not None:
        predictor.calibrate(baseline_data, feature_columns)
    
    return predictor


if __name__ == "__main__":
    # Test early warning predictor
    np.random.seed(42)
    
    # Generate baseline (healthy) data
    n_baseline = 100
    baseline = pd.DataFrame({
        "rms": np.random.normal(0.05, 0.01, n_baseline),
        "kurtosis": np.random.normal(0.5, 0.1, n_baseline),
        "temperature": np.random.normal(45, 2, n_baseline),
    })
    
    # Create and calibrate predictor
    predictor = EarlyWarningPredictor()
    predictor.calibrate(baseline)
    
    print("Testing Early Warning Predictor")
    print("=" * 50)
    
    # Simulate degrading data
    for i in range(50):
        # Gradually increasing values (degradation)
        sample = {
            "rms": 0.05 + i * 0.002 + np.random.normal(0, 0.005),
            "kurtosis": 0.5 + i * 0.03 + np.random.normal(0, 0.05),
            "temperature": 45 + i * 0.2 + np.random.normal(0, 1),
        }
        predictor.add_sample(sample)
    
    # Get prediction
    result = predictor.predict()
    
    print(f"Has Warning: {result.has_warning}")
    print(f"Earliest Anomaly: {result.earliest_anomaly_hours} hours")
    print(f"Risk Score: {result.overall_risk_score:.1f}")
    print(f"RUL: {result.remaining_useful_life_hours} hours")
    
    if result.predictions:
        print("\nPredictions:")
        for pred in result.predictions:
            print(f"  - {pred.affected_metrics[0]}: {pred.severity.value} in {pred.predicted_time_hours}h (confidence: {pred.confidence:.2f})")
            for rec in pred.recommendations:
                print(f"    → {rec}")
