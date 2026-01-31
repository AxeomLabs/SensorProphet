"""
Utility Functions for Predictive Maintenance System
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import LOGS_DIR


def setup_logging(name: str = "predictive_maintenance", level: int = logging.INFO) -> logging.Logger:
    """Set up logging with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate moving average of data."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def zscore(data: np.ndarray) -> np.ndarray:
    """Calculate z-scores for data array."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - mean) / std


def timestamp_to_datetime(timestamp_str: str) -> datetime:
    """Convert NASA dataset timestamp string to datetime."""
    # Format: 2004.02.12.10.32.39
    return datetime.strptime(timestamp_str, "%Y.%m.%d.%H.%M.%S")


def datetime_to_timestamp(dt: datetime) -> str:
    """Convert datetime to NASA dataset timestamp format."""
    return dt.strftime("%Y.%m.%d.%H.%M.%S")


def calculate_rms(signal: np.ndarray) -> float:
    """Calculate Root Mean Square of a signal."""
    return np.sqrt(np.mean(signal ** 2))


def calculate_kurtosis(signal: np.ndarray) -> float:
    """Calculate kurtosis of a signal."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return 0.0
    n = len(signal)
    return np.sum(((signal - mean) / std) ** 4) / n - 3


def calculate_peak_to_peak(signal: np.ndarray) -> float:
    """Calculate peak-to-peak amplitude of a signal."""
    return np.max(signal) - np.min(signal)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_health_score(score: float) -> str:
    """Format health score with status indicator."""
    if score >= 70:
        status = "Good"
    elif score >= 40:
        status = "Warning"
    else:
        status = "Critical"
    return f"{score:.1f}% ({status})"


def get_severity_level(health_score: float) -> str:
    """Get alert severity level based on health score."""
    if health_score >= 70:
        return "info"
    elif health_score >= 40:
        return "warning"
    return "critical"
