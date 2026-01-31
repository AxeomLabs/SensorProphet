"""
Time Series Forecasting Module

Implements multiple forecasting methods:
- ARIMA
- Prophet
- Rolling window analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

from src.config import FORECAST_HORIZONS
from src.utils import setup_logging

logger = setup_logging("forecaster")

# Import optional dependencies
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available.")


class TimeSeriesForecaster:
    """Multi-method time series forecaster for sensor data."""
    
    def __init__(
        self,
        method: str = "prophet",
        forecast_horizon: int = 24,
        confidence_level: float = 0.95,
    ):
        """
        Initialize forecaster.
        
        Args:
            method: Forecasting method ('prophet', 'arima', 'rolling').
            forecast_horizon: Number of steps ahead to forecast.
            confidence_level: Confidence level for prediction intervals.
        """
        self.method = method
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        
        self.model = None
        self.is_fitted = False
        self._history = None
        self._column_name = None
    
    def fit(
        self,
        data: Union[pd.Series, pd.DataFrame],
        column: Optional[str] = None,
    ) -> "TimeSeriesForecaster":
        """
        Fit the forecaster on historical data.
        
        Args:
            data: Time series data with datetime index.
            column: Column name if DataFrame.
            
        Returns:
            Self for chaining.
        """
        # Handle input
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            series = data[column]
        else:
            series = data
            column = data.name or "value"
        
        self._column_name = column
        self._history = series.copy()
        
        if self.method == "prophet":
            self._fit_prophet(series)
        elif self.method == "arima":
            self._fit_arima(series)
        elif self.method == "rolling":
            # Rolling method doesn't need fitting
            pass
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Fitted {self.method} forecaster on {len(series)} samples")
        return self
    
    def _fit_prophet(self, series: pd.Series) -> None:
        """Fit Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            "ds": series.index,
            "y": series.values,
        })
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=True,
                interval_width=self.confidence_level,
            )
            self.model.fit(df)
    
    def _fit_arima(self, series: pd.Series) -> None:
        """Fit ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is not installed")
        
        # Auto-select ARIMA order (simplified)
        # In production, use auto_arima from pmdarima
        order = (1, 1, 1)  # (p, d, q)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ARIMA(series, order=order)
            self.model = self.model.fit()
    
    def predict(
        self,
        horizon: Optional[int] = None,
        return_confidence: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            horizon: Number of steps to forecast.
            return_confidence: Whether to include confidence intervals.
            
        Returns:
            DataFrame with forecasts and optional confidence intervals.
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        horizon = horizon or self.forecast_horizon
        
        if self.method == "prophet":
            return self._predict_prophet(horizon, return_confidence)
        elif self.method == "arima":
            return self._predict_arima(horizon, return_confidence)
        elif self.method == "rolling":
            return self._predict_rolling(horizon, return_confidence)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _predict_prophet(self, horizon: int, return_confidence: bool) -> pd.DataFrame:
        """Generate Prophet forecasts."""
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon, freq='H')
        forecast = self.model.predict(future)
        
        # Get only future predictions
        result = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result.columns = ["timestamp", "forecast", "lower", "upper"]
        result.set_index("timestamp", inplace=True)
        
        if not return_confidence:
            result = result[["forecast"]]
        
        return result
    
    def _predict_arima(self, horizon: int, return_confidence: bool) -> pd.DataFrame:
        """Generate ARIMA forecasts."""
        forecast = self.model.get_forecast(steps=horizon)
        
        # Get predictions and confidence intervals
        mean = forecast.predicted_mean
        conf = forecast.conf_int(alpha=1 - self.confidence_level)
        
        # Create future timestamps
        last_time = self._history.index[-1]
        freq = pd.infer_freq(self._history.index) or 'H'
        future_times = pd.date_range(start=last_time, periods=horizon + 1, freq=freq)[1:]
        
        result = pd.DataFrame({
            "forecast": mean.values,
            "lower": conf.iloc[:, 0].values,
            "upper": conf.iloc[:, 1].values,
        }, index=future_times)
        
        if not return_confidence:
            result = result[["forecast"]]
        
        return result
    
    def _predict_rolling(self, horizon: int, return_confidence: bool) -> pd.DataFrame:
        """Generate rolling window forecasts (simple baseline)."""
        # Use last window mean and std for prediction
        window = min(24, len(self._history))
        recent = self._history.tail(window)
        
        mean_val = recent.mean()
        std_val = recent.std()
        
        # Create future timestamps
        last_time = self._history.index[-1]
        freq = pd.infer_freq(self._history.index) or 'H'
        future_times = pd.date_range(start=last_time, periods=horizon + 1, freq=freq)[1:]
        
        # Add trend based on recent change
        trend = (recent.iloc[-1] - recent.iloc[0]) / window
        forecasts = mean_val + trend * np.arange(1, horizon + 1)
        
        result = pd.DataFrame({
            "forecast": forecasts,
            "lower": forecasts - 2 * std_val,
            "upper": forecasts + 2 * std_val,
        }, index=future_times)
        
        if not return_confidence:
            result = result[["forecast"]]
        
        return result
    
    def evaluate(
        self,
        test_data: pd.Series,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate forecast accuracy on test data.
        
        Args:
            test_data: Actual values for comparison.
            metrics: List of metrics to compute.
            
        Returns:
            Dictionary of metric values.
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before evaluation")
        
        metrics = metrics or ["mae", "rmse", "mape"]
        horizon = len(test_data)
        
        # Generate forecasts
        forecast = self.predict(horizon=horizon, return_confidence=False)
        predictions = forecast["forecast"].values[:horizon]
        actuals = test_data.values[:horizon]
        
        results = {}
        
        if "mae" in metrics:
            results["mae"] = np.mean(np.abs(actuals - predictions))
        
        if "rmse" in metrics:
            results["rmse"] = np.sqrt(np.mean((actuals - predictions) ** 2))
        
        if "mape" in metrics:
            nonzero_mask = actuals != 0
            if np.any(nonzero_mask):
                results["mape"] = np.mean(
                    np.abs((actuals[nonzero_mask] - predictions[nonzero_mask]) / actuals[nonzero_mask])
                ) * 100
            else:
                results["mape"] = np.nan
        
        if "r2" in metrics:
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            results["r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return results


class MultiHorizonForecaster:
    """Forecast at multiple horizons simultaneously."""
    
    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        method: str = "prophet",
    ):
        """
        Initialize multi-horizon forecaster.
        
        Args:
            horizons: List of forecast horizons.
            method: Forecasting method.
        """
        self.horizons = horizons or FORECAST_HORIZONS
        self.method = method
        self.forecaster = None
    
    def fit(self, data: Union[pd.Series, pd.DataFrame], column: Optional[str] = None) -> "MultiHorizonForecaster":
        """Fit the forecaster."""
        self.forecaster = TimeSeriesForecaster(
            method=self.method,
            forecast_horizon=max(self.horizons),
        )
        self.forecaster.fit(data, column)
        return self
    
    def predict(self) -> Dict[int, pd.DataFrame]:
        """
        Generate forecasts for all horizons.
        
        Returns:
            Dictionary mapping horizon to forecast DataFrame.
        """
        results = {}
        for horizon in self.horizons:
            results[horizon] = self.forecaster.predict(horizon=horizon)
        return results


def forecast_sensor_metric(
    data: pd.DataFrame,
    column: str,
    horizon: int = 24,
    method: str = "prophet",
) -> pd.DataFrame:
    """
    Convenience function to forecast a sensor metric.
    
    Args:
        data: DataFrame with datetime index.
        column: Column to forecast.
        horizon: Forecast horizon.
        method: Forecasting method.
        
    Returns:
        DataFrame with forecasts.
    """
    forecaster = TimeSeriesForecaster(method=method, forecast_horizon=horizon)
    forecaster.fit(data, column)
    return forecaster.predict()


if __name__ == "__main__":
    # Test forecaster
    np.random.seed(42)
    
    # Generate test time series
    n_points = 200
    dates = pd.date_range(start="2024-01-01", periods=n_points, freq="H")
    
    # Create signal with trend and seasonality
    trend = np.linspace(0, 5, n_points)
    seasonality = 2 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    noise = np.random.randn(n_points) * 0.5
    values = 10 + trend + seasonality + noise
    
    series = pd.Series(values, index=dates, name="sensor_value")
    
    # Split train/test
    train = series[:-24]
    test = series[-24:]
    
    # Test with rolling method (always available)
    print("Testing rolling forecast:")
    forecaster = TimeSeriesForecaster(method="rolling", forecast_horizon=24)
    forecaster.fit(train)
    forecast = forecaster.predict()
    print(forecast.head())
    
    metrics = forecaster.evaluate(test)
    print(f"Metrics: {metrics}")
