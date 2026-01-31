"""
Predictive Maintenance System - Main Pipeline

Orchestrates data ingestion, processing, analysis, and visualization
for industrial equipment health monitoring.
"""
import asyncio
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Callable
import pandas as pd
import numpy as np

from src.config import (
    DATA_DIR, MODELS_DIR, SAMPLING_RATE, UPDATE_INTERVAL_MS,
    DASHBOARD_PORT, DASHBOARD_HOST
)
from src.utils import setup_logging

from src.data.data_loader import BearingDataLoader
from src.data.sensor_simulator import SensorSimulator
from src.data.mqtt_handler import get_mqtt_handler

from src.processing.preprocessor import DataPreprocessor
from src.processing.feature_extractor import FeatureExtractor

from src.anomaly.anomaly_detector import AnomalyDetector
from src.anomaly.autoencoder import get_autoencoder_detector

from src.models.forecaster import TimeSeriesForecaster
from src.models.health_scorer import HealthScorer

from src.alerts.alert_engine import AlertEngine

logger = setup_logging("pipeline")


class PredictiveMaintenancePipeline:
    """Main pipeline for predictive maintenance system."""
    
    def __init__(
        self,
        use_real_data: bool = True,
        data_dir: Optional[Path] = None,
        csv_file: Optional[Path] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            use_real_data: Whether to use real NASA dataset or simulated data.
            data_dir: Path to the data directory.
            csv_file: Path to a cleaned CSV file with bearing features.
        """
        self.use_real_data = use_real_data
        self.data_dir = data_dir or DATA_DIR
        self.csv_file = csv_file
        self.csv_data = None
        self.csv_index = 0
        
        # Components
        self.data_loader = None
        self.simulator = None
        self.mqtt_handler = None
        
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        self.anomaly_detector = AnomalyDetector(methods=["zscore", "isolation_forest"])
        self.health_scorer = HealthScorer()
        
        self.alert_engine = AlertEngine()
        
        # Data storage
        self.data_store = {
            "equipment": {},
            "alerts": [],
            "history": pd.DataFrame(),
            "features": pd.DataFrame(),
        }
        
        # Pipeline state
        self.is_running = False
        self._pipeline_thread = None
        self._stop_event = threading.Event()
        
        logger.info("Pipeline initialized")
    
    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing pipeline components...")
        
        # Try to load CSV file first
        if self.csv_file and Path(self.csv_file).exists():
            self.csv_data = pd.read_csv(self.csv_file)
            if 'timestamp' in self.csv_data.columns:
                self.csv_data['timestamp'] = pd.to_datetime(self.csv_data['timestamp'])
            logger.info(f"Loaded CSV data: {len(self.csv_data)} rows, columns: {list(self.csv_data.columns)}")
            self.use_real_data = True
        elif self.use_real_data:
            try:
                self.data_loader = BearingDataLoader(self.data_dir)
                logger.info(f"Loaded dataset: {self.data_loader.get_info()}")
            except FileNotFoundError:
                logger.warning("Dataset not found, using simulator")
                self.use_real_data = False
        
        if not self.use_real_data and self.csv_data is None:
            self.simulator = SensorSimulator(equipment_id="pump_001")
            logger.info("Using sensor simulator")
        
        # Initialize MQTT (using simulator for demo)
        self.mqtt_handler = get_mqtt_handler(use_simulator=True)
        self.mqtt_handler.connect()
        
        # Train models on baseline data
        self._train_models()
        
        # Register alert callback
        self.alert_engine.register_callback(self._on_alert)
        
        logger.info("Pipeline initialization complete")
    
    def _train_models(self) -> None:
        """Train anomaly detection and health models."""
        logger.info("Training models on baseline data...")
        
        if self.csv_data is not None:
            # Use first 100 rows as baseline from CSV
            baseline_df = self.csv_data.head(100).copy()
            numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Fit anomaly detector
            self.anomaly_detector.fit(baseline_df[numeric_cols])
            
            # Calibrate health scorer
            self.health_scorer.calibrate(baseline_df, numeric_cols)
            
            # Store feature columns for later use
            self._feature_cols = numeric_cols
            
        elif self.use_real_data and self.data_loader:
            # Use early portion of data as baseline
            baseline_df = self.data_loader.load_all(end_idx=100)
            
            # Preprocess
            numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Fit anomaly detector
            self.anomaly_detector.fit(baseline_df[numeric_cols])
            
            # Calibrate health scorer
            self.health_scorer.calibrate(baseline_df, numeric_cols)
            
            self._feature_cols = numeric_cols
        else:
            # Generate synthetic baseline data matching simulator output
            baseline_data = pd.DataFrame({
                "vibration": np.random.normal(0.05, 0.01, 100),
                "temperature": np.random.normal(45, 2, 100),
                "pressure": np.random.normal(100, 2, 100),
            })
            
            self.anomaly_detector.fit(baseline_data)
            self.health_scorer.calibrate(baseline_data)
            self._feature_cols = list(baseline_data.columns)
        
        logger.info("Model training complete")
    
    def _on_alert(self, alert) -> None:
        """Handle new alerts."""
        self.data_store["alerts"].insert(0, alert.to_dict())
        # Keep only recent alerts
        self.data_store["alerts"] = self.data_store["alerts"][:100]
    
    def process_sample(self, equipment_id: str, raw_data: Dict) -> Dict:
        """
        Process a single data sample through the pipeline.
        
        Args:
            equipment_id: Equipment identifier.
            raw_data: Raw sensor readings.
            
        Returns:
            Processing results dictionary.
        """
        timestamp = datetime.now().isoformat()
        
        # Extract features
        features = {}
        for key, value in raw_data.items():
            if isinstance(value, np.ndarray):
                # Extract features from raw signal
                signal_features = self.feature_extractor.extract_time_domain(value)
                for feat_name, feat_value in signal_features.items():
                    features[f"{key}_{feat_name}"] = feat_value
            else:
                features[key] = value
        
        # Prepare feature vector for anomaly detection
        feature_df = pd.DataFrame([features])
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Detect anomalies
        try:
            prediction = self.anomaly_detector.predict(feature_df[numeric_cols])
            is_anomaly = prediction[0] == -1
            anomaly_score = self.anomaly_detector.score_samples(feature_df[numeric_cols])[0]
        except Exception as e:
            logger.warning(f"Anomaly detection error: {e}")
            is_anomaly = False
            anomaly_score = 0.0
        
        # Calculate health score
        health_score = self.health_scorer.calculate_score(features, equipment_id)
        health_status = self.health_scorer.get_status(health_score)
        
        # Check alerts
        self.alert_engine.check_health_score(equipment_id, health_score)
        if is_anomaly:
            self.alert_engine.check_anomaly(equipment_id, is_anomaly, anomaly_score)
        self.alert_engine.check_thresholds(equipment_id, features)
        
        # Update data store
        self.data_store["equipment"][equipment_id] = {
            "health_score": health_score,
            "status": health_status.value,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "last_update": timestamp,
            "features": features,
        }
        
        return {
            "equipment_id": equipment_id,
            "timestamp": timestamp,
            "health_score": health_score,
            "status": health_status.value,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "features": features,
        }
    
    def run_batch_analysis(self, start_idx: int = 0, end_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Run batch analysis on historical data.
        
        Args:
            start_idx: Start index.
            end_idx: End index.
            
        Returns:
            DataFrame with analysis results.
        """
        if not self.use_real_data or not self.data_loader:
            logger.warning("Batch analysis requires real data")
            return pd.DataFrame()
        
        logger.info(f"Running batch analysis from {start_idx} to {end_idx}...")
        
        results = []
        
        for idx, (timestamp, signals) in enumerate(self.data_loader.stream_data()):
            if idx < start_idx:
                continue
            if end_idx and idx >= end_idx:
                break
            
            # Process sample
            result = self.process_sample("bearing_test", signals)
            result["file_idx"] = idx
            result["file_timestamp"] = timestamp
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} samples...")
        
        df = pd.DataFrame(results)
        logger.info(f"Batch analysis complete: {len(df)} samples processed")
        
        return df
    
    def start(self, interval_seconds: float = 1.0) -> None:
        """
        Start the pipeline for real-time processing.
        
        Args:
            interval_seconds: Interval between samples.
        """
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self._stop_event.clear()
        self.is_running = True
        
        def pipeline_loop():
            logger.info("Pipeline started")
            
            while not self._stop_event.is_set():
                try:
                    if self.csv_data is not None:
                        # Stream from CSV data
                        if self.csv_index >= len(self.csv_data):
                            self.csv_index = 0  # Loop back to start
                        
                        row = self.csv_data.iloc[self.csv_index]
                        self.csv_index += 1
                        
                        # Convert row to dict, excluding timestamp
                        sample = {}
                        for col in self._feature_cols:
                            if col in row.index:
                                sample[col] = float(row[col])
                        
                        equipment_id = "bearing_sensor"
                        
                        # Process
                        result = self.process_sample(equipment_id, sample)
                        
                        # Publish to MQTT
                        self.mqtt_handler.publish_sensor_data(equipment_id, sample)
                        
                    elif self.simulator:
                        # Get simulated data
                        sample = self.simulator.generate_sample()
                        equipment_id = sample.pop("equipment_id", "pump_001")
                        sample.pop("timestamp", None)
                        
                        # Process
                        result = self.process_sample(equipment_id, sample)
                        
                        # Publish to MQTT
                        self.mqtt_handler.publish_sensor_data(equipment_id, sample)
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    time.sleep(1)
            
            logger.info("Pipeline stopped")
        
        self._pipeline_thread = threading.Thread(target=pipeline_loop, daemon=True)
        self._pipeline_thread.start()
    
    def stop(self) -> None:
        """Stop the pipeline."""
        self._stop_event.set()
        self.is_running = False
        
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=5)
        
        logger.info("Pipeline stopped")
    
    def run_dashboard(self) -> None:
        """Start the dashboard."""
        from src.dashboard.app import run_dashboard
        run_dashboard(self.data_store)
    
    def get_summary(self) -> Dict:
        """Get current system summary."""
        return {
            "pipeline_running": self.is_running,
            "total_equipment": len(self.data_store["equipment"]),
            "active_alerts": len([a for a in self.data_store["alerts"] if not a.get("resolved")]),
            "alert_summary": self.alert_engine.get_alert_summary(),
            "equipment_status": {
                eq_id: {
                    "health_score": data.get("health_score"),
                    "status": data.get("status"),
                }
                for eq_id, data in self.data_store["equipment"].items()
            },
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predictive Maintenance System")
    parser.add_argument("--mode", choices=["demo", "batch", "dashboard"], default="demo",
                        help="Operating mode")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--csv-file", type=str, help="Path to cleaned CSV file with bearing features")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PredictiveMaintenancePipeline(
        use_real_data=not args.simulate,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        csv_file=Path(args.csv_file) if args.csv_file else None,
    )
    
    pipeline.initialize()
    
    if args.mode == "demo":
        # Run demo with simulated data
        logger.info("Starting demo mode...")
        pipeline.start(interval_seconds=1.0)
        
        print("\nPredictive Maintenance System - Demo Mode")
        print("=" * 50)
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                summary = pipeline.get_summary()
                print(f"\rEquipment health: ", end="")
                for eq_id, status in summary["equipment_status"].items():
                    print(f"{eq_id}: {status['health_score']:.0f}% ", end="")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            pipeline.stop()
    
    elif args.mode == "batch":
        # Run batch analysis
        logger.info("Starting batch analysis...")
        results = pipeline.run_batch_analysis()
        
        if not results.empty:
            output_file = "analysis_results.csv"
            results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
    
    elif args.mode == "dashboard":
        # Start dashboard
        pipeline.start(interval_seconds=1.0)
        pipeline.run_dashboard()


if __name__ == "__main__":
    main()
