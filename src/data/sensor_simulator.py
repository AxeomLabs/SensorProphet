"""
Sensor Data Simulator

Generates simulated sensor streams for testing and demonstration.
Supports vibration, temperature, and pressure sensors with configurable anomaly injection.
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Generator
import numpy as np

from src.config import SAMPLING_RATE
from src.utils import setup_logging

logger = setup_logging("sensor_simulator")


class SensorSimulator:
    """Generate simulated industrial sensor data."""
    
    def __init__(
        self,
        equipment_id: str = "pump_001",
        sensors: Optional[List[str]] = None,
        sampling_rate: int = 1,  # Hz for streaming
    ):
        """
        Initialize sensor simulator.
        
        Args:
            equipment_id: Identifier for the simulated equipment.
            sensors: List of sensor types to simulate.
            sampling_rate: Number of samples per second for streaming.
        """
        self.equipment_id = equipment_id
        self.sensors = sensors or ["vibration", "temperature", "pressure"]
        self.sampling_rate = sampling_rate
        
        # Base values for each sensor type
        self.base_values = {
            "vibration": 0.05,  # g
            "temperature": 45.0,  # Celsius
            "pressure": 100.0,  # PSI
        }
        
        # Noise levels
        self.noise_levels = {
            "vibration": 0.01,
            "temperature": 0.5,
            "pressure": 2.0,
        }
        
        # Degradation state
        self.degradation_factor = 0.0
        self.is_degrading = False
        self.sample_count = 0
        
        # Anomaly injection
        self.anomaly_active = False
        self.anomaly_type = None
        
    def _generate_base_signal(self, sensor_type: str, n_samples: int = 1) -> np.ndarray:
        """Generate base signal with noise."""
        base = self.base_values.get(sensor_type, 1.0)
        noise = self.noise_levels.get(sensor_type, 0.1)
        
        signal = base + np.random.normal(0, noise, n_samples)
        
        # Add degradation effect
        if self.is_degrading:
            degradation_noise = self.degradation_factor * base * 0.5
            signal += np.abs(np.random.normal(0, degradation_noise, n_samples))
        
        return signal
    
    def _inject_anomaly(self, signal: np.ndarray, sensor_type: str) -> np.ndarray:
        """Inject anomaly patterns into signal."""
        if not self.anomaly_active:
            return signal
        
        if self.anomaly_type == "spike":
            # Random spikes
            spike_idx = np.random.choice(len(signal), size=max(1, len(signal) // 10), replace=False)
            signal[spike_idx] *= 3.0
            
        elif self.anomaly_type == "drift":
            # Gradual drift upward
            drift = np.linspace(0, self.base_values[sensor_type] * 0.3, len(signal))
            signal += drift
            
        elif self.anomaly_type == "oscillation":
            # Add oscillatory pattern
            t = np.linspace(0, 2 * np.pi, len(signal))
            signal += np.sin(t * 5) * self.base_values[sensor_type] * 0.2
            
        elif self.anomaly_type == "failure":
            # Progressive failure pattern
            failure_factor = np.linspace(1, 3, len(signal))
            noise_increase = np.linspace(1, 5, len(signal))
            signal *= failure_factor
            signal += np.random.normal(0, self.noise_levels[sensor_type], len(signal)) * noise_increase
        
        return signal
    
    def generate_sample(self) -> Dict[str, float]:
        """
        Generate a single sample for all sensors.
        
        Returns:
            Dictionary mapping sensor names to values.
        """
        self.sample_count += 1
        
        # Update degradation
        if self.is_degrading:
            self.degradation_factor = min(1.0, self.degradation_factor + 0.001)
        
        sample = {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": self.equipment_id,
        }
        
        for sensor in self.sensors:
            signal = self._generate_base_signal(sensor, 1)
            signal = self._inject_anomaly(signal, sensor)
            sample[sensor] = float(signal[0])
        
        return sample
    
    def generate_batch(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate a batch of samples for all sensors.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Dictionary mapping sensor names to signal arrays.
        """
        batch = {}
        
        for sensor in self.sensors:
            signal = self._generate_base_signal(sensor, n_samples)
            signal = self._inject_anomaly(signal, sensor)
            batch[sensor] = signal
        
        return batch
    
    def stream(
        self,
        duration_seconds: Optional[float] = None,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> Generator[Dict[str, float], None, None]:
        """
        Stream sensor data continuously.
        
        Args:
            duration_seconds: Optional duration limit.
            callback: Optional callback function for each sample.
            
        Yields:
            Sensor readings as dictionaries.
        """
        start_time = time.time()
        interval = 1.0 / self.sampling_rate
        
        while True:
            sample = self.generate_sample()
            
            if callback:
                callback(sample)
            
            yield sample
            
            # Check duration limit
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break
            
            time.sleep(interval)
    
    def start_degradation(self) -> None:
        """Start simulating equipment degradation."""
        self.is_degrading = True
        self.degradation_factor = 0.0
        logger.info(f"Started degradation simulation for {self.equipment_id}")
    
    def stop_degradation(self) -> None:
        """Stop degradation simulation and reset."""
        self.is_degrading = False
        self.degradation_factor = 0.0
        logger.info(f"Stopped degradation simulation for {self.equipment_id}")
    
    def inject_anomaly(self, anomaly_type: str) -> None:
        """
        Start injecting anomalies.
        
        Args:
            anomaly_type: Type of anomaly ('spike', 'drift', 'oscillation', 'failure').
        """
        if anomaly_type not in ["spike", "drift", "oscillation", "failure"]:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        self.anomaly_active = True
        self.anomaly_type = anomaly_type
        logger.info(f"Injecting {anomaly_type} anomaly for {self.equipment_id}")
    
    def clear_anomaly(self) -> None:
        """Stop anomaly injection."""
        self.anomaly_active = False
        self.anomaly_type = None
        logger.info(f"Cleared anomaly for {self.equipment_id}")
    
    def simulate_bearing_failure(self, steps: int = 100) -> List[Dict]:
        """
        Simulate a complete bearing failure scenario.
        
        Args:
            steps: Number of time steps to simulate.
            
        Returns:
            List of sensor readings showing failure progression.
        """
        readings = []
        base_time = datetime.now()
        
        # Phase 1: Normal operation (30%)
        normal_steps = int(steps * 0.3)
        for i in range(normal_steps):
            sample = self.generate_sample()
            sample["timestamp"] = (base_time + timedelta(minutes=i * 10)).isoformat()
            sample["phase"] = "normal"
            readings.append(sample)
        
        # Phase 2: Early degradation (30%)
        self.start_degradation()
        degrade_steps = int(steps * 0.3)
        for i in range(degrade_steps):
            sample = self.generate_sample()
            sample["timestamp"] = (base_time + timedelta(minutes=(normal_steps + i) * 10)).isoformat()
            sample["phase"] = "degradation"
            readings.append(sample)
        
        # Phase 3: Failure approach (40%)
        self.inject_anomaly("failure")
        failure_steps = steps - normal_steps - degrade_steps
        for i in range(failure_steps):
            sample = self.generate_sample()
            sample["timestamp"] = (base_time + timedelta(minutes=(normal_steps + degrade_steps + i) * 10)).isoformat()
            sample["phase"] = "failure"
            readings.append(sample)
        
        # Reset state
        self.stop_degradation()
        self.clear_anomaly()
        
        return readings


class MultiEquipmentSimulator:
    """Simulate multiple pieces of equipment simultaneously."""
    
    def __init__(self, equipment_ids: List[str]):
        """
        Initialize multi-equipment simulator.
        
        Args:
            equipment_ids: List of equipment identifiers.
        """
        self.simulators = {
            eq_id: SensorSimulator(equipment_id=eq_id)
            for eq_id in equipment_ids
        }
    
    def generate_all(self) -> Dict[str, Dict[str, float]]:
        """Generate samples for all equipment."""
        return {
            eq_id: sim.generate_sample()
            for eq_id, sim in self.simulators.items()
        }
    
    def start_random_degradation(self, probability: float = 0.2) -> List[str]:
        """Start degradation on random equipment."""
        affected = []
        for eq_id, sim in self.simulators.items():
            if random.random() < probability:
                sim.start_degradation()
                affected.append(eq_id)
        return affected


if __name__ == "__main__":
    # Test the simulator
    simulator = SensorSimulator(equipment_id="test_pump")
    
    # Generate single sample
    print("Single sample:", simulator.generate_sample())
    
    # Generate batch
    batch = simulator.generate_batch(100)
    print("Batch shapes:", {k: v.shape for k, v in batch.items()})
    
    # Simulate failure
    print("\nSimulating failure scenario...")
    failure_data = simulator.simulate_bearing_failure(steps=20)
    for reading in failure_data[:5]:
        print(f"  {reading['phase']}: vibration={reading['vibration']:.4f}")
