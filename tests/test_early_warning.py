"""
Unit tests for early warning prediction module.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.early_warning import (
    EarlyWarningPredictor,
    EarlyWarningResult,
    PredictedAnomaly,
    PredictionSeverity,
    create_early_warning_predictor,
)


class TestEarlyWarningPredictor:
    """Tests for EarlyWarningPredictor class."""
    
    @pytest.fixture
    def baseline_data(self):
        """Generate baseline (healthy) data."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame({
            "rms": np.random.normal(0.05, 0.01, n_samples),
            "kurtosis": np.random.normal(0.5, 0.1, n_samples),
            "temperature": np.random.normal(45, 2, n_samples),
        })
    
    @pytest.fixture
    def calibrated_predictor(self, baseline_data):
        """Create a calibrated predictor."""
        predictor = EarlyWarningPredictor()
        predictor.calibrate(baseline_data)
        return predictor
    
    def test_predictor_initialization(self):
        """Test predictor can be initialized."""
        predictor = EarlyWarningPredictor()
        assert predictor is not None
        assert not predictor.is_calibrated
        assert predictor.horizons == [2, 6, 12, 24, 48]
    
    def test_predictor_calibration(self, baseline_data):
        """Test predictor calibration."""
        predictor = EarlyWarningPredictor()
        predictor.calibrate(baseline_data)
        
        assert predictor.is_calibrated
        assert "rms" in predictor._thresholds
        assert "kurtosis" in predictor._thresholds
        assert "temperature" in predictor._thresholds
    
    def test_add_sample(self, calibrated_predictor):
        """Test adding samples to history."""
        sample = {"rms": 0.05, "kurtosis": 0.5, "temperature": 45}
        calibrated_predictor.add_sample(sample)
        
        assert len(calibrated_predictor._history) == 1
        assert len(calibrated_predictor._timestamps) == 1
    
    def test_predict_without_enough_samples(self, calibrated_predictor):
        """Test prediction with insufficient samples."""
        # Add only a few samples (less than min_samples)
        for _ in range(5):
            sample = {"rms": 0.05, "kurtosis": 0.5, "temperature": 45}
            calibrated_predictor.add_sample(sample)
        
        result = calibrated_predictor.predict()
        
        assert isinstance(result, EarlyWarningResult)
        assert not result.has_warning
        assert result.predictions == []
    
    def test_predict_with_stable_data(self, calibrated_predictor):
        """Test prediction with stable (healthy) data."""
        np.random.seed(42)
        
        # Add stable samples around baseline
        for _ in range(50):
            sample = {
                "rms": 0.05 + np.random.normal(0, 0.005),
                "kurtosis": 0.5 + np.random.normal(0, 0.05),
                "temperature": 45 + np.random.normal(0, 1),
            }
            calibrated_predictor.add_sample(sample)
        
        result = calibrated_predictor.predict()
        
        assert isinstance(result, EarlyWarningResult)
        # Stable data should not trigger many warnings
        assert result.overall_risk_score < 50
    
    def test_predict_with_degrading_data(self, calibrated_predictor):
        """Test prediction with degrading data."""
        np.random.seed(42)
        
        # Add degrading samples (values increasing over time)
        for i in range(50):
            sample = {
                "rms": 0.05 + i * 0.003 + np.random.normal(0, 0.002),
                "kurtosis": 0.5 + i * 0.04 + np.random.normal(0, 0.02),
                "temperature": 45 + i * 0.3 + np.random.normal(0, 0.5),
            }
            calibrated_predictor.add_sample(sample)
        
        result = calibrated_predictor.predict()
        
        assert isinstance(result, EarlyWarningResult)
        # Degrading data should trigger warnings
        assert result.has_warning or result.overall_risk_score > 0
    
    def test_prediction_result_structure(self, calibrated_predictor):
        """Test that prediction results have correct structure."""
        for i in range(30):
            sample = {"rms": 0.05 + i * 0.002, "kurtosis": 0.5, "temperature": 45}
            calibrated_predictor.add_sample(sample)
        
        result = calibrated_predictor.predict()
        
        # Check result structure
        assert hasattr(result, "timestamp")
        assert hasattr(result, "has_warning")
        assert hasattr(result, "earliest_anomaly_hours")
        assert hasattr(result, "predictions")
        assert hasattr(result, "degradation_rate")
        assert hasattr(result, "remaining_useful_life_hours")
        assert hasattr(result, "overall_risk_score")
    
    def test_result_to_dict(self, calibrated_predictor):
        """Test result serialization to dictionary."""
        for i in range(30):
            sample = {"rms": 0.05, "kurtosis": 0.5, "temperature": 45}
            calibrated_predictor.add_sample(sample)
        
        result = calibrated_predictor.predict()
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "timestamp" in result_dict
        assert "has_warning" in result_dict
        assert "predictions" in result_dict
        assert isinstance(result_dict["predictions"], list)
    
    def test_reset(self, calibrated_predictor):
        """Test resetting history buffer."""
        for _ in range(10):
            sample = {"rms": 0.05, "kurtosis": 0.5, "temperature": 45}
            calibrated_predictor.add_sample(sample)
        
        assert len(calibrated_predictor._history) == 10
        
        calibrated_predictor.reset()
        
        assert len(calibrated_predictor._history) == 0
        assert len(calibrated_predictor._timestamps) == 0
    
    def test_history_summary(self, calibrated_predictor):
        """Test get_history_summary method."""
        summary = calibrated_predictor.get_history_summary()
        assert summary["samples"] == 0
        
        for _ in range(5):
            sample = {"rms": 0.05, "kurtosis": 0.5, "temperature": 45}
            calibrated_predictor.add_sample(sample)
        
        summary = calibrated_predictor.get_history_summary()
        assert summary["samples"] == 5
        assert "oldest" in summary
        assert "newest" in summary
        assert "features" in summary


class TestPredictedAnomaly:
    """Tests for PredictedAnomaly dataclass."""
    
    def test_predicted_anomaly_creation(self):
        """Test creating a PredictedAnomaly."""
        pred = PredictedAnomaly(
            predicted_time_hours=6.0,
            confidence=0.85,
            severity=PredictionSeverity.WARNING,
            affected_metrics=["rms"],
            current_values={"rms": 0.08},
            predicted_values={"rms": 0.15},
            thresholds={"rms": 0.12},
            recommendations=["Schedule maintenance"],
        )
        
        assert pred.predicted_time_hours == 6.0
        assert pred.confidence == 0.85
        assert pred.severity == PredictionSeverity.WARNING
    
    def test_predicted_anomaly_to_dict(self):
        """Test PredictedAnomaly serialization."""
        pred = PredictedAnomaly(
            predicted_time_hours=12.0,
            confidence=0.75,
            severity=PredictionSeverity.CRITICAL,
            affected_metrics=["temperature"],
            current_values={"temperature": 65},
            predicted_values={"temperature": 85},
            thresholds={"temperature": 70},
        )
        
        pred_dict = pred.to_dict()
        
        assert isinstance(pred_dict, dict)
        assert pred_dict["predicted_time_hours"] == 12.0
        assert pred_dict["severity"] == "critical"


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_without_baseline(self):
        """Test creating predictor without baseline data."""
        predictor = create_early_warning_predictor()
        
        assert predictor is not None
        assert not predictor.is_calibrated
    
    def test_create_with_baseline(self):
        """Test creating predictor with baseline data."""
        np.random.seed(42)
        baseline = pd.DataFrame({
            "metric_a": np.random.normal(10, 1, 50),
            "metric_b": np.random.normal(20, 2, 50),
        })
        
        predictor = create_early_warning_predictor(baseline)
        
        assert predictor.is_calibrated
        assert "metric_a" in predictor._thresholds
        assert "metric_b" in predictor._thresholds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
