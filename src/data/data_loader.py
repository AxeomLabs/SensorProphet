"""
NASA Bearing Dataset Loader

Loads and processes the NASA IMS Bearing Dataset (2nd test).
The dataset contains 984 vibration snapshots from a bearing degradation experiment.
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Generator
import numpy as np
import pandas as pd

from src.config import DATA_DIR, SENSOR_COLUMNS, SAMPLES_PER_FILE
from src.utils import setup_logging, timestamp_to_datetime

logger = setup_logging("data_loader")


class BearingDataLoader:
    """Load and process NASA IMS Bearing Dataset."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the dataset directory. Defaults to config.DATA_DIR.
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.file_list: List[Path] = []
        self.timestamps: List[datetime] = []
        self._scan_files()
    
    def _scan_files(self) -> None:
        """Scan directory for data files and extract timestamps."""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get all files (NASA dataset files are named with timestamps)
        files = sorted([f for f in self.data_dir.iterdir() if f.is_file()])
        
        for file_path in files:
            try:
                # Parse timestamp from filename (format: 2004.02.12.10.32.39)
                ts = timestamp_to_datetime(file_path.name)
                self.file_list.append(file_path)
                self.timestamps.append(ts)
            except ValueError:
                logger.warning(f"Skipping file with invalid timestamp: {file_path.name}")
        
        logger.info(f"Found {len(self.file_list)} data files spanning "
                   f"{self.timestamps[0]} to {self.timestamps[-1]}" if self.timestamps else "No files found")
    
    def load_file(self, file_path: Path) -> np.ndarray:
        """
        Load a single data file.
        
        Args:
            file_path: Path to the data file.
            
        Returns:
            2D numpy array with shape (samples, channels).
        """
        try:
            # NASA bearing data is tab-separated with 4 channels
            data = np.loadtxt(file_path, delimiter='\t')
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def load_all(self, start_idx: int = 0, end_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Load all data files and return as DataFrame.
        
        Args:
            start_idx: Start index of files to load.
            end_idx: End index of files to load (exclusive).
            
        Returns:
            DataFrame with timestamp index and bearing columns.
        """
        if end_idx is None:
            end_idx = len(self.file_list)
        
        records = []
        for i, (file_path, ts) in enumerate(zip(
            self.file_list[start_idx:end_idx],
            self.timestamps[start_idx:end_idx]
        )):
            if (i + 1) % 100 == 0:
                logger.info(f"Loading file {i + 1}/{end_idx - start_idx}")
            
            data = self.load_file(file_path)
            
            # Calculate summary statistics for each bearing
            record = {"timestamp": ts}
            for j, col in enumerate(SENSOR_COLUMNS):
                if j < data.shape[1]:
                    record[f"{col}_rms"] = np.sqrt(np.mean(data[:, j] ** 2))
                    record[f"{col}_peak"] = np.max(np.abs(data[:, j]))
                    record[f"{col}_mean"] = np.mean(data[:, j])
                    record[f"{col}_std"] = np.std(data[:, j])
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def load_raw_signals(self, file_index: int) -> Dict[str, np.ndarray]:
        """
        Load raw signals from a specific file.
        
        Args:
            file_index: Index of the file to load.
            
        Returns:
            Dictionary mapping bearing names to signal arrays.
        """
        if file_index >= len(self.file_list):
            raise IndexError(f"File index {file_index} out of range")
        
        data = self.load_file(self.file_list[file_index])
        return {
            col: data[:, i] for i, col in enumerate(SENSOR_COLUMNS)
            if i < data.shape[1]
        }
    
    def stream_data(self, delay_factor: float = 0.0) -> Generator[Tuple[datetime, Dict[str, np.ndarray]], None, None]:
        """
        Stream data files one by one (for simulation).
        
        Args:
            delay_factor: Optional delay between files (for real-time simulation).
            
        Yields:
            Tuple of (timestamp, signal_dict) for each file.
        """
        import time
        
        for file_path, ts in zip(self.file_list, self.timestamps):
            data = self.load_file(file_path)
            signals = {
                col: data[:, i] for i, col in enumerate(SENSOR_COLUMNS)
                if i < data.shape[1]
            }
            yield ts, signals
            
            if delay_factor > 0:
                time.sleep(delay_factor)
    
    def get_failure_region(self) -> Tuple[int, int]:
        """
        Get indices of the failure region in the dataset.
        
        The 2nd test shows bearing 1 outer race failure at the end.
        
        Returns:
            Tuple of (start_idx, end_idx) for failure region.
        """
        # Based on NASA dataset documentation, failure occurs in last ~100 samples
        total = len(self.file_list)
        return (total - 100, total)
    
    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            "total_files": len(self.file_list),
            "start_time": self.timestamps[0] if self.timestamps else None,
            "end_time": self.timestamps[-1] if self.timestamps else None,
            "duration_hours": (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 3600
                if len(self.timestamps) >= 2 else 0,
            "sampling_interval_minutes": 10,  # NASA dataset takes samples every 10 minutes
            "channels": SENSOR_COLUMNS,
        }


def load_processed_data(filepath: str = None) -> pd.DataFrame:
    """
    Load pre-processed bearing data if available.
    
    Args:
        filepath: Path to processed CSV file.
        
    Returns:
        DataFrame with processed data.
    """
    if filepath is None:
        filepath = r"c:\Users\test\Desktop\final\processed_bearing_data.csv"
    
    if not os.path.exists(filepath):
        logger.warning(f"Processed data file not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    logger.info(f"Loaded processed data: {len(df)} records")
    return df


if __name__ == "__main__":
    # Test the data loader
    loader = BearingDataLoader()
    print(loader.get_info())
    
    # Load first 10 files
    df = loader.load_all(end_idx=10)
    print(df.head())
