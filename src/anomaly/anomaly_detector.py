"""
Anomaly Detection Module

Implements multiple anomaly detection methods for industrial sensor data:
- Statistical thresholding (z-score, IQR)
- Isolation Forest
- One-class SVM
- Real-time scoring
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
import joblib
from pathlib import Path

from src.config import ANOMALY_THRESHOLDS, MODELS_DIR
from src.utils import setup_logging, zscore

logger = setup_logging("anomaly_detector")


class AnomalyDetector:
    """Multi-method anomaly detector for predictive maintenance."""
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        contamination: float = 0.1,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            methods: List of detection methods to use.
            contamination: Expected proportion of anomalies.
        """
        self.methods = methods or ["zscore", "isolation_forest"]
        self.contamination = contamination
        
        # Initialize models
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.thresholds: Dict = {}
        
        # Fit state
        self.is_fitted = False
        self._baseline_stats: Dict = {}
    
    def _init_models(self, n_features: int) -> None:
        """Initialize ML models."""
        if "isolation_forest" in self.methods:
            self.models["isolation_forest"] = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
        
        if "one_class_svm" in self.methods:
            self.models["one_class_svm"] = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,
                gamma='auto',
            )
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
    ) -> "AnomalyDetector":
        """
        Fit the anomaly detector on normal/baseline data.
        
        Args:
            data: Training data (should represent normal operation).
            feature_names: Names of features for reference.
            
        Returns:
            Self for chaining.
        """
        if isinstance(data, pd.DataFrame):
            feature_names = feature_names or data.columns.tolist()
            data = data.values
        
        n_samples, n_features = data.shape
        self._init_models(n_features)
        
        # Compute baseline statistics for z-score method
        self._baseline_stats = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
            "q1": np.percentile(data, 25, axis=0),
            "q3": np.percentile(data, 75, axis=0),
        }
        self._baseline_stats["iqr"] = self._baseline_stats["q3"] - self._baseline_stats["q1"]
        
        # Fit scaler
        self.scalers["standard"] = StandardScaler()
        data_scaled = self.scalers["standard"].fit_transform(data)
        
        # Fit models
        for method_name, model in self.models.items():
            logger.info(f"Fitting {method_name}...")
            model.fit(data_scaled)
        
        # Store thresholds
        self.thresholds = ANOMALY_THRESHOLDS.copy()
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        logger.info(f"Fitted anomaly detector on {n_samples} samples with {n_features} features")
        return self
    
    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            data: Data to check for anomalies.
            method: Specific method to use (None for ensemble).
            
        Returns:
            Binary array where 1 = normal, -1 = anomaly.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if method:
            return self._predict_single_method(data, method)
        
        # Ensemble prediction
        predictions = []
        for m in self.methods:
            pred = self._predict_single_method(data, m)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = np.sign(np.sum(predictions, axis=0))
        ensemble_pred[ensemble_pred == 0] = 1  # Tie goes to normal
        
        return ensemble_pred
    
    def _predict_single_method(self, data: np.ndarray, method: str) -> np.ndarray:
        """Predict using a single method."""
        if method == "zscore":
            return self._predict_zscore(data)
        elif method == "iqr":
            return self._predict_iqr(data)
        elif method in self.models:
            data_scaled = self.scalers["standard"].transform(data)
            return self.models[method].predict(data_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _predict_zscore(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using z-score method."""
        threshold = self.thresholds.get("rms_zscore", 3.0)
        
        mean = self._baseline_stats["mean"]
        std = self._baseline_stats["std"]
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        z_scores = np.abs((data - mean) / std)
        max_zscore = np.max(z_scores, axis=1)
        
        predictions = np.where(max_zscore > threshold, -1, 1)
        return predictions
    
    def _predict_iqr(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using IQR method."""
        q1 = self._baseline_stats["q1"]
        q3 = self._baseline_stats["q3"]
        iqr = self._baseline_stats["iqr"]
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        is_anomaly = np.any((data < lower) | (data > upper), axis=1)
        return np.where(is_anomaly, -1, 1)
    
    def score_samples(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: str = "isolation_forest",
    ) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).
        
        Args:
            data: Data to score.
            method: Scoring method.
            
        Returns:
            Array of anomaly scores.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if method == "isolation_forest" and "isolation_forest" in self.models:
            data_scaled = self.scalers["standard"].transform(data)
            return self.models["isolation_forest"].score_samples(data_scaled)
        elif method == "zscore":
            mean = self._baseline_stats["mean"]
            std = self._baseline_stats["std"]
            std = np.where(std == 0, 1, std)
            z_scores = np.abs((data - mean) / std)
            # Invert so higher score = more normal
            return -np.max(z_scores, axis=1)
        else:
            raise ValueError(f"Scoring not supported for method: {method}")
    
    def get_anomaly_report(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        timestamps: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Generate detailed anomaly report.
        
        Args:
            data: Data to analyze.
            timestamps: Optional timestamps for each sample.
            
        Returns:
            DataFrame with anomaly analysis results.
        """
        if isinstance(data, pd.DataFrame):
            timestamps = timestamps or data.index.tolist()
            data = data.values
        
        results = []
        
        for i in range(len(data)):
            sample = data[i:i+1]
            
            record = {
                "index": i,
                "timestamp": timestamps[i] if timestamps else i,
            }
            
            # Score with each method
            for method in self.methods:
                pred = self._predict_single_method(sample, method)
                record[f"{method}_prediction"] = pred[0]
            
            # Get isolation forest score if available
            if "isolation_forest" in self.models:
                record["anomaly_score"] = self.score_samples(sample, "isolation_forest")[0]
            
            # Ensemble prediction
            record["ensemble_prediction"] = self.predict(sample)[0]
            record["is_anomaly"] = record["ensemble_prediction"] == -1
            
            results.append(record)
        
        return pd.DataFrame(results)
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save the fitted detector to disk."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before saving")
        
        filepath = filepath or MODELS_DIR / "anomaly_detector.joblib"
        
        save_dict = {
            "methods": self.methods,
            "contamination": self.contamination,
            "models": self.models,
            "scalers": self.scalers,
            "thresholds": self.thresholds,
            "baseline_stats": self._baseline_stats,
            "feature_names": self.feature_names,
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Saved anomaly detector to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "AnomalyDetector":
        """Load a fitted detector from disk."""
        save_dict = joblib.load(filepath)
        
        detector = cls(
            methods=save_dict["methods"],
            contamination=save_dict["contamination"],
        )
        detector.models = save_dict["models"]
        detector.scalers = save_dict["scalers"]
        detector.thresholds = save_dict["thresholds"]
        detector._baseline_stats = save_dict["baseline_stats"]
        detector.feature_names = save_dict["feature_names"]
        detector.is_fitted = True
        
        logger.info(f"Loaded anomaly detector from {filepath}")
        return detector


class RealTimeAnomalyMonitor:
    """Real-time anomaly monitoring with windowed analysis."""
    
    def __init__(
        self,
        detector: AnomalyDetector,
        window_size: int = 10,
        alert_threshold: float = 0.5,
    ):
        """
        Initialize real-time monitor.
        
        Args:
            detector: Fitted anomaly detector.
            window_size: Number of samples to consider for trending.
            alert_threshold: Proportion of anomalies in window to trigger alert.
        """
        self.detector = detector
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Buffer for recent samples
        self.sample_buffer: List[np.ndarray] = []
        self.prediction_buffer: List[int] = []
        self.score_buffer: List[float] = []
    
    def process_sample(self, sample: np.ndarray) -> Dict:
        """
        Process a single sample in real-time.
        
        Args:
            sample: Single sample array.
            
        Returns:
            Dictionary with prediction results and alerts.
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Get prediction and score
        prediction = self.detector.predict(sample)[0]
        
        try:
            score = self.detector.score_samples(sample)[0]
        except:
            score = 0.0
        
        # Update buffers
        self.sample_buffer.append(sample)
        self.prediction_buffer.append(prediction)
        self.score_buffer.append(score)
        
        # Keep buffer at window size
        if len(self.sample_buffer) > self.window_size:
            self.sample_buffer.pop(0)
            self.prediction_buffer.pop(0)
            self.score_buffer.pop(0)
        
        # Calculate window statistics
        anomaly_rate = sum(1 for p in self.prediction_buffer if p == -1) / len(self.prediction_buffer)
        avg_score = np.mean(self.score_buffer)
        
        # Determine alert status
        should_alert = anomaly_rate >= self.alert_threshold
        
        return {
            "is_anomaly": prediction == -1,
            "anomaly_score": score,
            "window_anomaly_rate": anomaly_rate,
            "window_avg_score": avg_score,
            "alert": should_alert,
            "alert_level": self._get_alert_level(anomaly_rate, avg_score),
        }
    
    def _get_alert_level(self, anomaly_rate: float, avg_score: float) -> str:
        """Determine alert severity level."""
        if anomaly_rate >= 0.8:
            return "critical"
        elif anomaly_rate >= 0.5:
            return "warning"
        elif anomaly_rate >= 0.2:
            return "info"
        return "normal"
    
    def reset(self) -> None:
        """Reset the monitor buffers."""
        self.sample_buffer.clear()
        self.prediction_buffer.clear()
        self.score_buffer.clear()


if __name__ == "__main__":
    # Test anomaly detector
    np.random.seed(42)
    
    # Generate normal data
    n_normal = 500
    normal_data = np.random.randn(n_normal, 5) * 0.5 + 2
    
    # Generate anomalous data
    n_anomaly = 50
    anomaly_data = np.random.randn(n_anomaly, 5) * 2 + 5
    
    # Combine
    test_data = np.vstack([normal_data, anomaly_data])
    labels = np.array([1] * n_normal + [-1] * n_anomaly)
    
    # Train on normal data only
    detector = AnomalyDetector(methods=["zscore", "isolation_forest"])
    detector.fit(normal_data)
    
    # Predict
    predictions = detector.predict(test_data)
    
    # Calculate accuracy
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["Anomaly", "Normal"]))
    
    # Test real-time monitor
    print("\nReal-time monitoring test:")
    monitor = RealTimeAnomalyMonitor(detector, window_size=10)
    
    for i in range(20):
        sample = test_data[i]
        result = monitor.process_sample(sample)
        if result["alert"]:
            print(f"  Sample {i}: ALERT - {result['alert_level']}")
