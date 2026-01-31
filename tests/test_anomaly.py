"""
Unit tests for anomaly detection module.
"""
import pytest
import numpy as np
import pandas as pd

from src.anomaly.anomaly_detector import AnomalyDetector, RealTimeAnomalyMonitor
from src.anomaly.autoencoder import SimpleAutoencoder, get_autoencoder_detector


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""
    
    @pytest.fixture
    def sample_normal_data(self):
        """Generate normal baseline data."""
        np.random.seed(42)
        return np.random.randn(100, 5) * 0.5 + 2
    
    @pytest.fixture
    def sample_anomaly_data(self):
        """Generate anomalous data."""
        np.random.seed(42)
        return np.random.randn(20, 5) * 2 + 5
    
    @pytest.fixture
    def fitted_detector(self, sample_normal_data):
        """Create a fitted detector."""
        detector = AnomalyDetector(methods=["zscore", "isolation_forest"])
        detector.fit(sample_normal_data)
        return detector
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = AnomalyDetector()
        assert detector is not None
        assert not detector.is_fitted
    
    def test_detector_fit(self, sample_normal_data):
        """Test detector fitting."""
        detector = AnomalyDetector()
        detector.fit(sample_normal_data)
        
        assert detector.is_fitted
        assert "mean" in detector._baseline_stats
        assert "std" in detector._baseline_stats
    
    def test_detector_predict(self, fitted_detector, sample_normal_data, sample_anomaly_data):
        """Test anomaly prediction."""
        # Normal data should mostly be predicted as normal
        normal_pred = fitted_detector.predict(sample_normal_data)
        normal_rate = np.mean(normal_pred == 1)
        assert normal_rate > 0.8, f"Normal detection rate too low: {normal_rate}"
        
        # Anomaly data should mostly be predicted as anomalous
        anomaly_pred = fitted_detector.predict(sample_anomaly_data)
        anomaly_rate = np.mean(anomaly_pred == -1)
        assert anomaly_rate > 0.5, f"Anomaly detection rate too low: {anomaly_rate}"
    
    def test_detector_score_samples(self, fitted_detector, sample_normal_data, sample_anomaly_data):
        """Test anomaly scoring."""
        normal_scores = fitted_detector.score_samples(sample_normal_data)
        anomaly_scores = fitted_detector.score_samples(sample_anomaly_data)
        
        # Anomalies should have lower scores (more negative)
        assert np.mean(normal_scores) > np.mean(anomaly_scores)
    
    def test_single_sample_prediction(self, fitted_detector):
        """Test prediction on single sample."""
        sample = np.array([2.0, 2.1, 1.9, 2.0, 2.0])
        pred = fitted_detector.predict(sample)
        
        assert len(pred) == 1
        assert pred[0] in [-1, 1]
    
    def test_unfitted_detector_raises(self):
        """Test that unfitted detector raises error."""
        detector = AnomalyDetector()
        
        with pytest.raises(ValueError):
            detector.predict(np.array([[1, 2, 3, 4, 5]]))


class TestRealTimeMonitor:
    """Tests for RealTimeAnomalyMonitor class."""
    
    @pytest.fixture
    def monitor(self, fitted_detector):
        """Create a monitor."""
        return RealTimeAnomalyMonitor(fitted_detector, window_size=10)
    
    @pytest.fixture
    def fitted_detector(self):
        """Create a fitted detector."""
        np.random.seed(42)
        normal_data = np.random.randn(100, 5) * 0.5 + 2
        detector = AnomalyDetector(methods=["zscore"])
        detector.fit(normal_data)
        return detector
    
    def test_monitor_process_sample(self, monitor):
        """Test processing single sample."""
        sample = np.array([2.0, 2.1, 1.9, 2.0, 2.0])
        result = monitor.process_sample(sample)
        
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "window_anomaly_rate" in result
        assert "alert" in result
    
    def test_monitor_window_tracking(self, monitor):
        """Test that window is tracked correctly."""
        # Process 15 samples
        for _ in range(15):
            sample = np.random.randn(5) * 0.5 + 2
            monitor.process_sample(sample)
        
        # Buffer should be at window size
        assert len(monitor.sample_buffer) == 10
        assert len(monitor.prediction_buffer) == 10
    
    def test_monitor_reset(self, monitor):
        """Test monitor reset."""
        for _ in range(5):
            sample = np.random.randn(5) * 0.5 + 2
            monitor.process_sample(sample)
        
        monitor.reset()
        
        assert len(monitor.sample_buffer) == 0
        assert len(monitor.prediction_buffer) == 0


class TestSimpleAutoencoder:
    """Tests for SimpleAutoencoder class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        return np.random.randn(100, 10) * 0.5 + 2
    
    def test_autoencoder_fit(self, sample_data):
        """Test autoencoder fitting."""
        ae = SimpleAutoencoder(n_components=5)
        ae.fit(sample_data)
        
        assert ae.is_fitted
        assert ae.mean is not None
        assert ae.threshold is not None
    
    def test_autoencoder_predict(self, sample_data):
        """Test autoencoder prediction."""
        ae = SimpleAutoencoder(n_components=5)
        ae.fit(sample_data)
        
        # Normal data
        pred = ae.predict(sample_data[:10])
        normal_rate = np.mean(pred == 1)
        assert normal_rate > 0.5
        
        # Anomaly data
        anomaly_data = np.random.randn(10, 10) * 3 + 10
        pred = ae.predict(anomaly_data)
        anomaly_rate = np.mean(pred == -1)
        assert anomaly_rate > 0.3
    
    def test_autoencoder_reconstruction_error(self, sample_data):
        """Test reconstruction error calculation."""
        ae = SimpleAutoencoder(n_components=5)
        ae.fit(sample_data)
        
        errors = ae.get_reconstruction_error(sample_data[:10])
        
        assert len(errors) == 10
        assert all(e >= 0 for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
