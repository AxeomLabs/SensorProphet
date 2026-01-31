"""
Data Preprocessing Module

Handles data cleaning, normalization, noise filtering, and missing value imputation
for industrial sensor data.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Optional, Tuple, Union

from src.utils import setup_logging

logger = setup_logging("preprocessor")


class DataPreprocessor:
    """Preprocess sensor data for anomaly detection and prediction."""
    
    def __init__(
        self,
        scaling_method: str = "standard",
        filter_type: str = "butterworth",
        filter_cutoff: float = 0.1,
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'.
            filter_type: 'butterworth', 'moving_average', or 'none'.
            filter_cutoff: Cutoff frequency for filters (0-1, normalized).
        """
        self.scaling_method = scaling_method
        self.filter_type = filter_type
        self.filter_cutoff = filter_cutoff
        
        # Initialize scaler
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        self.is_fitted = False
        self._fit_params = {}
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "DataPreprocessor":
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Training data.
            
        Returns:
            Self for chaining.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.scaler.fit(data)
        self.is_fitted = True
        
        # Store statistics for reference
        self._fit_params = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
        }
        
        logger.info(f"Fitted preprocessor with {data.shape[0]} samples")
        return self
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Data to transform.
            
        Returns:
            Transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        return self.scaler.transform(data)
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def impute_missing(
        data: Union[np.ndarray, pd.DataFrame],
        method: str = "forward_fill",
        max_gap: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Impute missing values in data.
        
        Args:
            data: Data with potential missing values.
            method: 'forward_fill', 'backward_fill', 'interpolate', 'mean', 'median'.
            max_gap: Maximum consecutive NaN gap to fill.
            
        Returns:
            Data with missing values imputed.
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        if not is_dataframe:
            data = pd.DataFrame(data)
        
        original_nan_count = data.isna().sum().sum()
        
        if method == "forward_fill":
            data = data.ffill(limit=max_gap)
            data = data.bfill(limit=max_gap)  # Fill remaining at start
        elif method == "backward_fill":
            data = data.bfill(limit=max_gap)
            data = data.ffill(limit=max_gap)
        elif method == "interpolate":
            data = data.interpolate(method='linear', limit=max_gap)
            data = data.bfill(limit=max_gap)
            data = data.ffill(limit=max_gap)
        elif method == "mean":
            data = data.fillna(data.mean())
        elif method == "median":
            data = data.fillna(data.median())
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        final_nan_count = data.isna().sum().sum()
        if original_nan_count > 0:
            logger.info(f"Imputed {original_nan_count - final_nan_count} missing values")
        
        return data if is_dataframe else data.values
    
    def filter_noise(
        self,
        signal_data: np.ndarray,
        method: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply noise filtering to signal.
        
        Args:
            signal_data: Input signal.
            method: Override default filter type.
            **kwargs: Additional filter parameters.
            
        Returns:
            Filtered signal.
        """
        method = method or self.filter_type
        
        if method == "none":
            return signal_data
        
        if method == "butterworth":
            return self._butterworth_filter(signal_data, **kwargs)
        elif method == "moving_average":
            return self._moving_average_filter(signal_data, **kwargs)
        elif method == "savgol":
            return self._savgol_filter(signal_data, **kwargs)
        elif method == "median":
            return self._median_filter(signal_data, **kwargs)
        else:
            raise ValueError(f"Unknown filter type: {method}")
    
    def _butterworth_filter(
        self,
        signal_data: np.ndarray,
        order: int = 4,
        cutoff: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Butterworth lowpass filter."""
        cutoff = cutoff or self.filter_cutoff
        
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff, btype='low')
        
        # Apply filter
        return signal.filtfilt(b, a, signal_data, axis=0)
    
    def _moving_average_filter(
        self,
        signal_data: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """Apply moving average filter."""
        if signal_data.ndim == 1:
            kernel = np.ones(window_size) / window_size
            return np.convolve(signal_data, kernel, mode='same')
        else:
            result = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                kernel = np.ones(window_size) / window_size
                result[:, i] = np.convolve(signal_data[:, i], kernel, mode='same')
            return result
    
    def _savgol_filter(
        self,
        signal_data: np.ndarray,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter."""
        if window_length % 2 == 0:
            window_length += 1
        
        if signal_data.ndim == 1:
            return signal.savgol_filter(signal_data, window_length, polyorder)
        else:
            result = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                result[:, i] = signal.savgol_filter(
                    signal_data[:, i], window_length, polyorder
                )
            return result
    
    def _median_filter(
        self,
        signal_data: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Apply median filter."""
        from scipy.ndimage import median_filter
        
        if signal_data.ndim == 1:
            return median_filter(signal_data, size=kernel_size)
        else:
            return median_filter(signal_data, size=(kernel_size, 1))
    
    @staticmethod
    def remove_outliers(
        data: np.ndarray,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and replace outliers.
        
        Args:
            data: Input data.
            method: 'zscore' or 'iqr'.
            threshold: Threshold for outlier detection.
            
        Returns:
            Tuple of (cleaned_data, outlier_mask).
        """
        if method == "zscore":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            z_scores = np.abs((data - mean) / std)
            outlier_mask = z_scores > threshold
            
        elif method == "iqr":
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_mask = (data < lower) | (data > upper)
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        # Replace outliers with median
        cleaned_data = data.copy()
        for i in range(data.shape[1] if data.ndim > 1 else 1):
            if data.ndim > 1:
                col_mask = outlier_mask[:, i]
                if col_mask.any():
                    cleaned_data[col_mask, i] = np.median(data[:, i])
            else:
                if outlier_mask.any():
                    cleaned_data[outlier_mask] = np.median(data)
        
        outlier_count = np.sum(outlier_mask)
        if outlier_count > 0:
            logger.info(f"Replaced {outlier_count} outliers")
        
        return cleaned_data, outlier_mask
    
    @staticmethod
    def resample(
        data: pd.DataFrame,
        target_freq: str,
        method: str = "mean",
    ) -> pd.DataFrame:
        """
        Resample time series data.
        
        Args:
            data: DataFrame with datetime index.
            target_freq: Target frequency ('1T', '5T', '1H', etc.).
            method: Aggregation method ('mean', 'max', 'min', 'sum').
            
        Returns:
            Resampled DataFrame.
        """
        if method == "mean":
            return data.resample(target_freq).mean()
        elif method == "max":
            return data.resample(target_freq).max()
        elif method == "min":
            return data.resample(target_freq).min()
        elif method == "sum":
            return data.resample(target_freq).sum()
        else:
            raise ValueError(f"Unknown resample method: {method}")


def preprocess_bearing_data(
    df: pd.DataFrame,
    scaling: bool = True,
    impute: bool = True,
    remove_outliers: bool = False,
) -> pd.DataFrame:
    """
    Convenience function for preprocessing bearing data.
    
    Args:
        df: Raw bearing data DataFrame.
        scaling: Whether to apply scaling.
        impute: Whether to impute missing values.
        remove_outliers: Whether to remove outliers.
        
    Returns:
        Preprocessed DataFrame.
    """
    preprocessor = DataPreprocessor(scaling_method="standard")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = df.copy()
    
    # Impute missing values
    if impute:
        result[numeric_cols] = DataPreprocessor.impute_missing(
            result[numeric_cols], method="interpolate"
        )
    
    # Remove outliers
    if remove_outliers:
        cleaned, _ = DataPreprocessor.remove_outliers(
            result[numeric_cols].values, method="zscore", threshold=4.0
        )
        result[numeric_cols] = cleaned
    
    # Scale data
    if scaling:
        result[numeric_cols] = preprocessor.fit_transform(result[numeric_cols])
    
    return result


if __name__ == "__main__":
    # Test preprocessor
    np.random.seed(42)
    
    # Generate test data with noise and outliers
    n_samples = 1000
    data = np.random.randn(n_samples, 4)
    data[100, 0] = 100  # Add outlier
    data[500:510, :] = np.nan  # Add missing values
    
    df = pd.DataFrame(data, columns=["ch1", "ch2", "ch3", "ch4"])
    
    print("Original data shape:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_imputed = DataPreprocessor.impute_missing(df, method="interpolate")
    df_cleaned, outliers = DataPreprocessor.remove_outliers(df_imputed.values)
    
    print("After imputation - Missing:", pd.DataFrame(df_imputed).isna().sum().sum())
    print("Outliers found:", np.sum(outliers))
