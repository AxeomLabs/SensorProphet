"""
MQTT Handler for Sensor Data Ingestion

Provides MQTT client functionality for publishing and subscribing to sensor data streams.
"""
import json
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any
from queue import Queue

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from src.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPICS
from src.utils import setup_logging

logger = setup_logging("mqtt_handler")


class MQTTHandler:
    """Handle MQTT connections for sensor data streaming."""
    
    def __init__(
        self,
        broker: str = MQTT_BROKER,
        port: int = MQTT_PORT,
        client_id: Optional[str] = None,
    ):
        """
        Initialize MQTT handler.
        
        Args:
            broker: MQTT broker address.
            port: MQTT broker port.
            client_id: Optional client identifier.
        """
        if not MQTT_AVAILABLE:
            logger.warning("paho-mqtt not installed. MQTT functionality unavailable.")
            self.client = None
            return
        
        self.broker = broker
        self.port = port
        self.client_id = client_id or f"predictive_maintenance_{int(time.time())}"
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_queue = Queue()
        
        # Connection state
        self.is_connected = False
        self._reconnect_thread = None
    
    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection callback."""
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
            
            # Re-subscribe to topics
            for topic in self.message_handlers.keys():
                self.client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection callback."""
        self.is_connected = False
        if rc != 0:
            logger.warning(f"Unexpected disconnection. Code: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming message callback."""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Add to queue
            self.message_queue.put((topic, payload))
            
            # Call registered handlers
            for handler_topic, handlers in self.message_handlers.items():
                if mqtt.topic_matches_sub(handler_topic, topic):
                    for handler in handlers:
                        try:
                            handler(topic, payload)
                        except Exception as e:
                            logger.error(f"Handler error for {topic}: {e}")
                            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to the MQTT broker.
        
        Args:
            timeout: Connection timeout in seconds.
            
        Returns:
            True if connected successfully.
        """
        if not self.client:
            logger.error("MQTT client not available")
            return False
        
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection
            start = time.time()
            while not self.is_connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from MQTT broker")
    
    def subscribe(self, topic: str, handler: Callable[[str, Dict], None]) -> None:
        """
        Subscribe to a topic with a message handler.
        
        Args:
            topic: MQTT topic pattern.
            handler: Callback function(topic, payload).
        """
        if topic not in self.message_handlers:
            self.message_handlers[topic] = []
            if self.is_connected:
                self.client.subscribe(topic)
        
        self.message_handlers[topic].append(handler)
        logger.info(f"Added handler for topic: {topic}")
    
    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """
        Publish a message to a topic.
        
        Args:
            topic: MQTT topic.
            payload: Message payload as dictionary.
            qos: Quality of service level.
            retain: Whether to retain the message.
            
        Returns:
            True if published successfully.
        """
        if not self.client or not self.is_connected:
            logger.warning("Cannot publish: not connected")
            return False
        
        try:
            message = json.dumps(payload)
            result = self.client.publish(topic, message, qos=qos, retain=retain)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return False
    
    def publish_sensor_data(
        self,
        equipment_id: str,
        sensor_data: Dict[str, float],
    ) -> bool:
        """
        Publish sensor data for an equipment.
        
        Args:
            equipment_id: Equipment identifier.
            sensor_data: Dictionary of sensor readings.
            
        Returns:
            True if published successfully.
        """
        topic = f"equipment/{equipment_id}/sensors"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": equipment_id,
            **sensor_data,
        }
        return self.publish(topic, payload)
    
    def publish_alert(
        self,
        equipment_id: str,
        alert_type: str,
        severity: str,
        message: str,
    ) -> bool:
        """
        Publish an alert for an equipment.
        
        Args:
            equipment_id: Equipment identifier.
            alert_type: Type of alert.
            severity: Alert severity level.
            message: Alert message.
            
        Returns:
            True if published successfully.
        """
        topic = f"equipment/{equipment_id}/alerts"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": equipment_id,
            "type": alert_type,
            "severity": severity,
            "message": message,
        }
        return self.publish(topic, payload)
    
    def publish_health_score(
        self,
        equipment_id: str,
        health_score: float,
        components: Dict[str, float],
    ) -> bool:
        """
        Publish health score for an equipment.
        
        Args:
            equipment_id: Equipment identifier.
            health_score: Overall health score (0-100).
            components: Individual component scores.
            
        Returns:
            True if published successfully.
        """
        topic = f"equipment/{equipment_id}/health"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": equipment_id,
            "health_score": health_score,
            "components": components,
        }
        return self.publish(topic, payload)


class InMemoryMQTTSimulator:
    """
    In-memory MQTT simulator for testing without a real broker.
    Mimics basic publish/subscribe functionality.
    """
    
    def __init__(self):
        """Initialize in-memory MQTT simulator."""
        self.topics: Dict[str, List[Callable]] = {}
        self.messages: List[Dict] = []
        self.is_connected = True
        logger.info("Using in-memory MQTT simulator")
    
    def connect(self, timeout: float = 10.0) -> bool:
        """Simulate connection."""
        self.is_connected = True
        return True
    
    def disconnect(self) -> None:
        """Simulate disconnection."""
        self.is_connected = False
    
    def subscribe(self, topic: str, handler: Callable[[str, Dict], None]) -> None:
        """Subscribe to a topic."""
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(handler)
    
    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """Publish a message and notify subscribers."""
        message = {"topic": topic, "payload": payload, "timestamp": datetime.now()}
        self.messages.append(message)
        
        # Notify matching subscribers
        for sub_topic, handlers in self.topics.items():
            if self._topic_matches(sub_topic, topic):
                for handler in handlers:
                    try:
                        handler(topic, payload)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
        
        return True
    
    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches pattern (simplified matching)."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        
        for p_part, t_part in zip(pattern_parts, topic_parts):
            if p_part == "#":
                return True
            if p_part == "+":
                continue
            if p_part != t_part:
                return False
        
        return len(pattern_parts) == len(topic_parts) or pattern_parts[-1] == "#"
    
    def publish_sensor_data(self, equipment_id: str, sensor_data: Dict[str, float]) -> bool:
        """Publish sensor data."""
        topic = f"equipment/{equipment_id}/sensors"
        payload = {"timestamp": datetime.now().isoformat(), "equipment_id": equipment_id, **sensor_data}
        return self.publish(topic, payload)
    
    def get_messages(self, topic_filter: Optional[str] = None) -> List[Dict]:
        """Get stored messages, optionally filtered by topic."""
        if topic_filter:
            return [m for m in self.messages if self._topic_matches(topic_filter, m["topic"])]
        return self.messages


def get_mqtt_handler(use_simulator: bool = True) -> MQTTHandler:
    """
    Get an MQTT handler instance.
    
    Args:
        use_simulator: If True, use in-memory simulator instead of real MQTT.
        
    Returns:
        MQTT handler instance.
    """
    if use_simulator or not MQTT_AVAILABLE:
        return InMemoryMQTTSimulator()
    return MQTTHandler()


if __name__ == "__main__":
    # Test with in-memory simulator
    handler = get_mqtt_handler(use_simulator=True)
    
    # Subscribe to sensor data
    received_messages = []
    def on_sensor_data(topic, payload):
        received_messages.append(payload)
        print(f"Received: {topic} -> {payload}")
    
    handler.subscribe("equipment/+/sensors", on_sensor_data)
    
    # Publish test data
    handler.publish_sensor_data("pump_001", {"vibration": 0.05, "temperature": 45.0})
    handler.publish_sensor_data("motor_002", {"vibration": 0.08, "temperature": 52.0})
    
    print(f"\nReceived {len(received_messages)} messages")
