"""
Unit tests for data processing module.
"""
import pytest
import numpy as np
import pandas as pd

from src.processing.preprocessor import DataPreprocessor, preprocess_bearing_data
from src.processing.feature_extractor import FeatureExtractor, extract_features_from_raw


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data with noise."""
        np.random.seed(42)
        return np.random.randn(100, 4)
    
    @pytest.fixture
    def data_with_missing(self):
        """Generate data with missing values."""
        np.random.seed(42)
        data = np.random.randn(100, 4)
        data[10:15, 0] = np.nan
        data[50, 2] = np.nan
        return pd.DataFrame(data, columns=["ch1", "ch2", "ch3", "ch4"])
    
    @pytest.fixture
    def data_with_outliers(self):
        """Generate data with outliers."""
        np.random.seed(42)
        data = np.random.randn(100, 4)
        data[50, 0] = 100  # Outlier
        data[75, 2] = -50  # Outlier
        return data
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        prep = DataPreprocessor()
        assert not prep.is_fitted
        
        prep = DataPreprocessor(scaling_method="minmax")
        assert prep.scaling_method == "minmax"
    
    def test_preprocessor_fit_transform(self, sample_data):
        """Test fitting and transforming."""
        prep = DataPreprocessor()
        transformed = prep.fit_transform(sample_data)
        
        assert prep.is_fitted
        assert transformed.shape == sample_data.shape
        
        # Check standardization (mean ~0, std ~1)
        assert abs(np.mean(transformed)) < 0.1
        assert abs(np.std(transformed) - 1) < 0.1
    
    def test_preprocessor_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        prep = DataPreprocessor()
        transformed = prep.fit_transform(sample_data)
        recovered = prep.inverse_transform(transformed)
        
        np.testing.assert_array_almost_equal(sample_data, recovered, decimal=10)
    
    def test_impute_missing_forward_fill(self, data_with_missing):
        """Test forward fill imputation."""
        result = DataPreprocessor.impute_missing(data_with_missing, method="forward_fill")
        
        assert not result.isna().any().any()
    
    def test_impute_missing_interpolate(self, data_with_missing):
        """Test interpolation imputation."""
        result = DataPreprocessor.impute_missing(data_with_missing, method="interpolate")
        
        assert not result.isna().any().any()
    
    def test_remove_outliers_zscore(self, data_with_outliers):
        """Test outlier removal using z-score."""
        cleaned, mask = DataPreprocessor.remove_outliers(
            data_with_outliers, method="zscore", threshold=3.0
        )
        
        assert mask.any()  # Some outliers should be detected
        assert cleaned[50, 0] != 100  # Outlier should be replaced
        assert cleaned[75, 2] != -50  # Outlier should be replaced
    
    def test_remove_outliers_iqr(self, data_with_outliers):
        """Test outlier removal using IQR."""
        cleaned, mask = DataPreprocessor.remove_outliers(
            data_with_outliers, method="iqr", threshold=1.5
        )
        
        assert mask.any()  # Some outliers should be detected
    
    def test_filter_noise_butterworth(self, sample_data):
        """Test Butterworth filter."""
        prep = DataPreprocessor()
        filtered = prep.filter_noise(sample_data[:, 0], method="butterworth")
        
        assert len(filtered) == len(sample_data[:, 0])
        # Filtered signal should have less variance (smoother)
        assert np.std(filtered) <= np.std(sample_data[:, 0])
    
    def test_filter_noise_moving_average(self, sample_data):
        """Test moving average filter."""
        prep = DataPreprocessor()
        filtered = prep.filter_noise(sample_data[:, 0], method="moving_average", window_size=5)
        
        assert len(filtered) == len(sample_data[:, 0])


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    @pytest.fixture
    def sample_signal(self):
        """Generate sample vibration signal."""
        np.random.seed(42)
        t = np.linspace(0, 1, 20480)
        # Sinusoidal signal with noise
        signal = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
        return signal
    
    @pytest.fixture
    def multi_channel_signals(self):
        """Generate multi-channel signals."""
        np.random.seed(42)
        n_samples = 20480
        return {
            "bearing1": np.random.randn(n_samples) * 0.1,
            "bearing2": np.random.randn(n_samples) * 0.15,
            "bearing3": np.random.randn(n_samples) * 0.12,
        }
    
    def test_extractor_init(self):
        """Test extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.sampling_rate > 0
    
    def test_extract_time_domain(self, sample_signal):
        """Test time-domain feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_time_domain(sample_signal)
        
        expected_features = ["mean", "std", "rms", "kurtosis", "skewness", 
                           "crest_factor", "peak_to_peak"]
        
        for feat in expected_features:
            assert feat in features, f"Missing feature: {feat}"
            assert not np.isnan(features[feat]), f"NaN value for: {feat}"
    
    def test_extract_frequency_domain(self, sample_signal):
        """Test frequency-domain feature extraction."""
        extractor = FeatureExtractor(sampling_rate=20480)
        features = extractor.extract_frequency_domain(sample_signal)
        
        expected_features = ["total_power", "dominant_freq", "spectral_centroid"]
        
        for feat in expected_features:
            assert feat in features, f"Missing feature: {feat}"
    
    def test_extract_envelope_features(self, sample_signal):
        """Test envelope feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_envelope_features(sample_signal)
        
        assert "envelope_mean" in features
        assert "envelope_std" in features
        assert "envelope_rms" in features
    
    def test_extract_all_features(self, sample_signal):
        """Test extraction of all features."""
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(sample_signal, "test")
        
        # Should have many features
        assert len(features) > 20
        
        # All features should be prefixed
        assert all(k.startswith("test_") for k in features.keys())
    
    def test_extract_multi_channel(self, multi_channel_signals):
        """Test multi-channel feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_multi_channel(multi_channel_signals)
        
        # Should have features for each channel
        assert any("bearing1" in k for k in features.keys())
        assert any("bearing2" in k for k in features.keys())
        assert any("bearing3" in k for k in features.keys())
        
        # Should have cross-channel features
        assert "cross_channel_rms_std" in features
    
    def test_extract_rolling_features(self):
        """Test rolling feature extraction."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        df = pd.DataFrame({
            "sensor1": np.random.randn(100),
            "sensor2": np.random.randn(100),
        }, index=dates)
        
        extractor = FeatureExtractor()
        result = extractor.extract_rolling_features(df, window=10)
        
        assert "sensor1_rolling_mean" in result.columns
        assert "sensor1_rolling_std" in result.columns
        assert "sensor1_diff" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
