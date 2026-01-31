"""
Feature Extraction Module

Extracts statistical and frequency-domain features from industrial sensor signals
for anomaly detection and health monitoring.
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Tuple, Union

from src.config import WINDOW_SIZE, SAMPLING_RATE, ROLLING_WINDOW
from src.utils import setup_logging, calculate_rms, calculate_kurtosis, calculate_peak_to_peak

logger = setup_logging("feature_extractor")


class FeatureExtractor:
    """Extract features from sensor signals for predictive maintenance."""
    
    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        window_size: int = WINDOW_SIZE,
    ):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz.
            window_size: Window size for FFT and rolling features.
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.feature_names: List[str] = []
    
    def extract_time_domain(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain statistical features.
        
        Args:
            signal_data: 1D signal array.
            
        Returns:
            Dictionary of time-domain features.
        """
        features = {}
        
        # Basic statistics
        features["mean"] = np.mean(signal_data)
        features["std"] = np.std(signal_data)
        features["var"] = np.var(signal_data)
        features["min"] = np.min(signal_data)
        features["max"] = np.max(signal_data)
        features["median"] = np.median(signal_data)
        
        # Root Mean Square
        features["rms"] = calculate_rms(signal_data)
        
        # Peak-to-Peak
        features["peak_to_peak"] = calculate_peak_to_peak(signal_data)
        
        # Crest Factor
        features["crest_factor"] = features["max"] / features["rms"] if features["rms"] > 0 else 0
        
        # Shape Factor
        features["shape_factor"] = features["rms"] / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) > 0 else 0
        
        # Impulse Factor
        features["impulse_factor"] = features["max"] / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) > 0 else 0
        
        # Higher-order statistics
        features["skewness"] = stats.skew(signal_data)
        features["kurtosis"] = calculate_kurtosis(signal_data)
        
        # Percentiles
        features["p25"] = np.percentile(signal_data, 25)
        features["p75"] = np.percentile(signal_data, 75)
        features["iqr"] = features["p75"] - features["p25"]
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features["zero_crossing_rate"] = len(zero_crossings) / len(signal_data)
        
        # Mean absolute deviation
        features["mad"] = np.mean(np.abs(signal_data - features["mean"]))
        
        # Energy
        features["energy"] = np.sum(signal_data ** 2)
        
        return features
    
    def extract_frequency_domain(
        self,
        signal_data: np.ndarray,
        n_freq_bands: int = 5,
    ) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Args:
            signal_data: 1D signal array.
            n_freq_bands: Number of frequency bands to analyze.
            
        Returns:
            Dictionary of frequency-domain features.
        """
        features = {}
        
        # Compute FFT
        n = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(n, 1 / self.sampling_rate)
        
        # Get positive frequencies only
        positive_mask = xf > 0
        xf_pos = xf[positive_mask]
        yf_pos = np.abs(yf[positive_mask])
        
        if len(yf_pos) == 0:
            return features
        
        # Normalize FFT magnitude
        yf_normalized = yf_pos / n
        power_spectrum = yf_normalized ** 2
        
        # Total spectral power
        features["total_power"] = np.sum(power_spectrum)
        
        # Dominant frequency
        dominant_idx = np.argmax(yf_pos)
        features["dominant_freq"] = xf_pos[dominant_idx]
        features["dominant_amplitude"] = yf_normalized[dominant_idx]
        
        # Spectral centroid (center of mass of spectrum)
        if features["total_power"] > 0:
            features["spectral_centroid"] = np.sum(xf_pos * power_spectrum) / features["total_power"]
        else:
            features["spectral_centroid"] = 0
        
        # Spectral spread
        if features["total_power"] > 0:
            features["spectral_spread"] = np.sqrt(
                np.sum(((xf_pos - features["spectral_centroid"]) ** 2) * power_spectrum) / features["total_power"]
            )
        else:
            features["spectral_spread"] = 0
        
        # Spectral entropy
        power_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        features["spectral_entropy"] = -np.sum(power_normalized * np.log2(power_normalized + 1e-10))
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        features["spectral_flatness"] = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Frequency band powers
        max_freq = self.sampling_rate / 2
        band_edges = np.linspace(0, max_freq, n_freq_bands + 1)
        
        for i in range(n_freq_bands):
            band_mask = (xf_pos >= band_edges[i]) & (xf_pos < band_edges[i + 1])
            if np.any(band_mask):
                features[f"band_{i + 1}_power"] = np.sum(power_spectrum[band_mask])
            else:
                features[f"band_{i + 1}_power"] = 0
        
        # Harmonic analysis - find peaks
        peaks, _ = signal.find_peaks(yf_normalized, height=np.max(yf_normalized) * 0.1)
        features["n_spectral_peaks"] = len(peaks)
        
        return features
    
    def extract_envelope_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract envelope-based features (useful for bearing fault detection).
        
        Args:
            signal_data: 1D signal array.
            
        Returns:
            Dictionary of envelope features.
        """
        features = {}
        
        # Compute envelope using Hilbert transform
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        
        # Envelope statistics
        features["envelope_mean"] = np.mean(envelope)
        features["envelope_std"] = np.std(envelope)
        features["envelope_max"] = np.max(envelope)
        features["envelope_rms"] = np.sqrt(np.mean(envelope ** 2))
        
        # Envelope spectrum features
        env_fft = np.abs(fft(envelope))
        env_freq = fftfreq(len(envelope), 1 / self.sampling_rate)
        
        positive_mask = env_freq > 0
        if np.any(positive_mask):
            env_fft_pos = env_fft[positive_mask]
            env_freq_pos = env_freq[positive_mask]
            
            features["envelope_dominant_freq"] = env_freq_pos[np.argmax(env_fft_pos)]
            features["envelope_total_power"] = np.sum(env_fft_pos ** 2)
        
        return features
    
    def extract_all_features(
        self,
        signal_data: np.ndarray,
        channel_name: str = "",
    ) -> Dict[str, float]:
        """
        Extract all features from a signal.
        
        Args:
            signal_data: 1D signal array.
            channel_name: Optional prefix for feature names.
            
        Returns:
            Dictionary of all features.
        """
        features = {}
        prefix = f"{channel_name}_" if channel_name else ""
        
        # Time domain features
        time_features = self.extract_time_domain(signal_data)
        for name, value in time_features.items():
            features[f"{prefix}{name}"] = value
        
        # Frequency domain features
        freq_features = self.extract_frequency_domain(signal_data)
        for name, value in freq_features.items():
            features[f"{prefix}{name}"] = value
        
        # Envelope features
        env_features = self.extract_envelope_features(signal_data)
        for name, value in env_features.items():
            features[f"{prefix}{name}"] = value
        
        return features
    
    def extract_multi_channel(
        self,
        signals: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Extract features from multiple channels.
        
        Args:
            signals: Dictionary mapping channel names to signal arrays.
            
        Returns:
            Dictionary of all features from all channels.
        """
        all_features = {}
        
        for channel_name, signal_data in signals.items():
            channel_features = self.extract_all_features(signal_data, channel_name)
            all_features.update(channel_features)
        
        # Add cross-channel features
        if len(signals) > 1:
            channel_rms = [calculate_rms(s) for s in signals.values()]
            all_features["cross_channel_rms_std"] = np.std(channel_rms)
            all_features["cross_channel_rms_max"] = np.max(channel_rms)
            all_features["cross_channel_rms_min"] = np.min(channel_rms)
        
        return all_features
    
    def extract_rolling_features(
        self,
        df: pd.DataFrame,
        window: int = ROLLING_WINDOW,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract rolling window features from time series data.
        
        Args:
            df: DataFrame with time series data.
            window: Rolling window size.
            feature_cols: Columns to compute rolling features for.
            
        Returns:
            DataFrame with rolling features added.
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = df.copy()
        
        for col in feature_cols:
            # Rolling statistics
            result[f"{col}_rolling_mean"] = df[col].rolling(window).mean()
            result[f"{col}_rolling_std"] = df[col].rolling(window).std()
            result[f"{col}_rolling_min"] = df[col].rolling(window).min()
            result[f"{col}_rolling_max"] = df[col].rolling(window).max()
            
            # Rate of change
            result[f"{col}_diff"] = df[col].diff()
            result[f"{col}_pct_change"] = df[col].pct_change()
            
            # Exponential moving average
            result[f"{col}_ema"] = df[col].ewm(span=window).mean()
            
            # Z-score relative to rolling window
            rolling_mean = df[col].rolling(window).mean()
            rolling_std = df[col].rolling(window).std()
            result[f"{col}_zscore"] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
        
        return result


def extract_features_from_raw(
    raw_signals: Dict[str, np.ndarray],
    sampling_rate: int = SAMPLING_RATE,
) -> Dict[str, float]:
    """
    Convenience function to extract features from raw signals.
    
    Args:
        raw_signals: Dictionary of channel name to signal array.
        sampling_rate: Sampling rate in Hz.
        
    Returns:
        Dictionary of extracted features.
    """
    extractor = FeatureExtractor(sampling_rate=sampling_rate)
    return extractor.extract_multi_channel(raw_signals)


def create_feature_dataframe(
    feature_list: List[Dict[str, float]],
    timestamps: Optional[List] = None,
) -> pd.DataFrame:
    """
    Create DataFrame from list of feature dictionaries.
    
    Args:
        feature_list: List of feature dictionaries.
        timestamps: Optional list of timestamps.
        
    Returns:
        DataFrame with features.
    """
    df = pd.DataFrame(feature_list)
    
    if timestamps:
        df.insert(0, "timestamp", timestamps)
        df.set_index("timestamp", inplace=True)
    
    return df


if __name__ == "__main__":
    # Test feature extraction
    np.random.seed(42)
    
    # Generate test signal (simulated vibration with a fault component)
    t = np.linspace(0, 1, 20480)
    signal_clean = np.sin(2 * np.pi * 100 * t)  # 100 Hz fundamental
    signal_fault = 0.3 * np.sin(2 * np.pi * 350 * t)  # Fault frequency
    noise = 0.1 * np.random.randn(len(t))
    test_signal = signal_clean + signal_fault + noise
    
    # Extract features
    extractor = FeatureExtractor(sampling_rate=20000)
    features = extractor.extract_all_features(test_signal, "bearing1")
    
    print("Extracted features:")
    for name, value in list(features.items())[:20]:
        print(f"  {name}: {value:.6f}")
    
    print(f"\nTotal features extracted: {len(features)}")
