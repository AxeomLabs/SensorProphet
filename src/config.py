"""
Centralized Configuration for Predictive Maintenance System
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(r"c:\Users\test\Desktop\final\2nd_test")
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data Configuration
SENSOR_COLUMNS = ["bearing1", "bearing2", "bearing3", "bearing4"]
SAMPLING_RATE = 20000  # Hz - NASA bearing dataset sampling rate
SAMPLES_PER_FILE = 20480  # Number of samples per snapshot

# Feature Extraction
WINDOW_SIZE = 2048  # Samples for FFT
ROLLING_WINDOW = 10  # Number of snapshots for rolling features

# Anomaly Detection Thresholds
ANOMALY_THRESHOLDS = {
    "rms_zscore": 3.0,
    "kurtosis_zscore": 3.0,
    "isolation_forest_contamination": 0.1,
    "autoencoder_percentile": 95,
}

# Health Score Configuration
HEALTH_SCORE_WEIGHTS = {
    "rms": 0.3,
    "kurtosis": 0.25,
    "peak_to_peak": 0.2,
    "fft_energy": 0.25,
}

# Prediction Models
FORECAST_HORIZONS = [2, 6, 12, 24, 48]  # Hours
LSTM_SEQUENCE_LENGTH = 50
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Early Warning Configuration
EARLY_WARNING_HORIZONS = [2, 6, 12, 24, 48]  # Hours to predict ahead
EARLY_WARNING_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to trigger warning
EARLY_WARNING_UPDATE_INTERVAL = 10  # Update predictions every N samples

# Alert Configuration
ALERT_LEVELS = {
    "info": {"health_min": 70, "color": "#17a2b8"},
    "warning": {"health_min": 40, "color": "#ffc107"},
    "critical": {"health_min": 0, "color": "#dc3545"},
}

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPICS = {
    "sensor_data": "equipment/+/sensors",
    "alerts": "equipment/+/alerts",
    "health": "equipment/+/health",
}

# Dashboard Configuration
DASHBOARD_HOST = "0.0.0.0"  # Accessible on local network
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = False  # Disable debug in network mode
UPDATE_INTERVAL_MS = 2000  # Refresh interval in milliseconds
