"""
Autoencoder-based Anomaly Detection

Uses reconstruction error from autoencoders to detect anomalies.
Works well for complex, multivariate sensor data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

from src.config import ANOMALY_THRESHOLDS, MODELS_DIR
from src.utils import setup_logging

logger = setup_logging("autoencoder")

# Try to import TensorFlow, fall back to simple implementation if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Using simple autoencoder implementation.")


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detector using reconstruction error."""
    
    def __init__(
        self,
        encoding_dims: Optional[List[int]] = None,
        activation: str = "relu",
        threshold_percentile: float = 95.0,
    ):
        """
        Initialize autoencoder detector.
        
        Args:
            encoding_dims: List of dimensions for encoder layers.
            activation: Activation function for hidden layers.
            threshold_percentile: Percentile of training errors for threshold.
        """
        self.encoding_dims = encoding_dims or [32, 16, 8]
        self.activation = activation
        self.threshold_percentile = threshold_percentile
        
        self.model = None
        self.encoder = None
        self.threshold = None
        self.input_dim = None
        self.is_fitted = False
        
        self._training_history = None
        self._mean = None
        self._std = None
    
    def _build_model(self, input_dim: int) -> None:
        """Build the autoencoder model."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using simple reconstruction.")
            return
        
        self.input_dim = input_dim
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        for dim in self.encoding_dims:
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Bottleneck
        bottleneck = x
        
        # Decoder (mirror of encoder)
        for dim in reversed(self.encoding_dims[:-1]):
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(input_dim, activation='linear')(x)
        
        # Create models
        self.model = Model(inputs, outputs, name="autoencoder")
        self.encoder = Model(inputs, bottleneck, name="encoder")
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
        )
        
        logger.info(f"Built autoencoder: {input_dim} -> {self.encoding_dims} -> {input_dim}")
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 0,
    ) -> "AutoencoderAnomalyDetector":
        """
        Train the autoencoder on normal data.
        
        Args:
            data: Training data (should represent normal operation).
            epochs: Number of training epochs.
            batch_size: Training batch size.
            validation_split: Fraction of data for validation.
            verbose: Verbosity level.
            
        Returns:
            Self for chaining.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Normalize data
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)
        self._std = np.where(self._std == 0, 1, self._std)
        
        data_normalized = (data - self._mean) / self._std
        
        if TF_AVAILABLE:
            # Build and train model
            self._build_model(data.shape[1])
            
            self._training_history = self.model.fit(
                data_normalized,
                data_normalized,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                    ),
                ],
            )
            
            # Calculate reconstruction errors on training data
            reconstructed = self.model.predict(data_normalized, verbose=0)
            errors = np.mean(np.square(data_normalized - reconstructed), axis=1)
        else:
            # Simple fallback: use mean as "reconstruction"
            self.input_dim = data.shape[1]
            errors = np.mean(np.square(data_normalized), axis=1)
        
        # Set threshold based on percentile of training errors
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        self.is_fitted = True
        logger.info(f"Trained autoencoder. Threshold: {self.threshold:.6f}")
        
        return self
    
    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            data: Data to check.
            
        Returns:
            Binary array where 1 = normal, -1 = anomaly.
        """
        errors = self.get_reconstruction_error(data)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions
    
    def get_reconstruction_error(
        self,
        data: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Calculate reconstruction error for data.
        
        Args:
            data: Input data.
            
        Returns:
            Array of reconstruction errors.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Normalize
        data_normalized = (data - self._mean) / self._std
        
        if TF_AVAILABLE and self.model is not None:
            reconstructed = self.model.predict(data_normalized, verbose=0)
            errors = np.mean(np.square(data_normalized - reconstructed), axis=1)
        else:
            # Fallback
            errors = np.mean(np.square(data_normalized), axis=1)
        
        return errors
    
    def get_anomaly_score(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get normalized anomaly scores (0 = normal, 1 = highly anomalous).
        
        Args:
            data: Input data.
            
        Returns:
            Array of normalized anomaly scores.
        """
        errors = self.get_reconstruction_error(data)
        
        # Normalize relative to threshold
        scores = errors / (self.threshold * 2)
        scores = np.clip(scores, 0, 1)
        
        return scores
    
    def get_latent_representation(
        self,
        data: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Get latent space representation from encoder.
        
        Args:
            data: Input data.
            
        Returns:
            Latent representations.
        """
        if not TF_AVAILABLE or self.encoder is None:
            raise ValueError("Encoder not available")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        data_normalized = (data - self._mean) / self._std
        return self.encoder.predict(data_normalized, verbose=0)
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = filepath or MODELS_DIR / "autoencoder"
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save model if TensorFlow available
        if TF_AVAILABLE and self.model is not None:
            self.model.save(filepath / "model.keras")
        
        # Save parameters
        params = {
            "encoding_dims": self.encoding_dims,
            "activation": self.activation,
            "threshold_percentile": self.threshold_percentile,
            "threshold": float(self.threshold),
            "input_dim": self.input_dim,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }
        
        with open(filepath / "params.json", 'w') as f:
            json.dump(params, f)
        
        logger.info(f"Saved autoencoder to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "AutoencoderAnomalyDetector":
        """Load a model from disk."""
        filepath = Path(filepath)
        
        # Load parameters
        with open(filepath / "params.json", 'r') as f:
            params = json.load(f)
        
        detector = cls(
            encoding_dims=params["encoding_dims"],
            activation=params["activation"],
            threshold_percentile=params["threshold_percentile"],
        )
        
        detector.threshold = params["threshold"]
        detector.input_dim = params["input_dim"]
        detector._mean = np.array(params["mean"])
        detector._std = np.array(params["std"])
        
        # Load model if available
        if TF_AVAILABLE and (filepath / "model.keras").exists():
            detector.model = keras.models.load_model(filepath / "model.keras")
            # Rebuild encoder
            detector._build_model(detector.input_dim)
        
        detector.is_fitted = True
        logger.info(f"Loaded autoencoder from {filepath}")
        
        return detector


class SimpleAutoencoder:
    """Simple PCA-based autoencoder for when TensorFlow is not available."""
    
    def __init__(self, n_components: int = 8, threshold_percentile: float = 95.0):
        """
        Initialize simple autoencoder.
        
        Args:
            n_components: Number of principal components.
            threshold_percentile: Percentile for anomaly threshold.
        """
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        
        self.components = None
        self.mean = None
        self.threshold = None
        self.is_fitted = False
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "SimpleAutoencoder":
        """Fit the simple autoencoder using PCA."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Center data
        self.mean = np.mean(data, axis=0)
        centered = data - self.mean
        
        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Keep top k components
        k = min(self.n_components, Vt.shape[0])
        self.components = Vt[:k]
        
        # Calculate reconstruction errors
        projected = centered @ self.components.T
        reconstructed = projected @ self.components
        errors = np.mean((centered - reconstructed) ** 2, axis=1)
        
        self.threshold = np.percentile(errors, self.threshold_percentile)
        self.is_fitted = True
        
        return self
    
    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict anomalies."""
        errors = self.get_reconstruction_error(data)
        return np.where(errors > self.threshold, -1, 1)
    
    def get_reconstruction_error(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get reconstruction errors."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        centered = data - self.mean
        projected = centered @ self.components.T
        reconstructed = projected @ self.components
        
        return np.mean((centered - reconstructed) ** 2, axis=1)


def get_autoencoder_detector(use_tensorflow: bool = True) -> Union[AutoencoderAnomalyDetector, SimpleAutoencoder]:
    """
    Get an autoencoder detector.
    
    Args:
        use_tensorflow: Whether to use TensorFlow if available.
        
    Returns:
        Autoencoder detector instance.
    """
    if use_tensorflow and TF_AVAILABLE:
        return AutoencoderAnomalyDetector()
    return SimpleAutoencoder()


if __name__ == "__main__":
    # Test autoencoder
    np.random.seed(42)
    
    # Generate normal data
    n_normal = 500
    normal_data = np.random.randn(n_normal, 10) * 0.5 + 2
    
    # Generate anomalous data
    n_anomaly = 50
    anomaly_data = np.random.randn(n_anomaly, 10) * 2 + 5
    
    # Train
    detector = get_autoencoder_detector(use_tensorflow=True)
    detector.fit(normal_data, epochs=20, verbose=0)
    
    # Test
    test_data = np.vstack([normal_data[:50], anomaly_data])
    predictions = detector.predict(test_data)
    
    print(f"Normal samples detected as normal: {np.sum(predictions[:50] == 1)}/50")
    print(f"Anomaly samples detected as anomaly: {np.sum(predictions[50:] == -1)}/50")
