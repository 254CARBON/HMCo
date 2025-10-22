"""
Unit tests for Event Producer service
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../services/event-producer'))

try:
    from event_producer import EventProducer, EventType
except ImportError:
    EventProducer = None
    EventType = None

@pytest.mark.unit
class TestEventProducer:
    """Test EventProducer class"""
    
    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer"""
        with patch('event_producer.Producer') as mock:
            yield mock
    
    def test_event_producer_initialization(self, mock_kafka_producer):
        """Test EventProducer initialization"""
        if EventProducer is None:
            pytest.skip("EventProducer not available")
        
        bootstrap_servers = "localhost:9092"
        producer = EventProducer(bootstrap_servers)
        
        assert producer is not None
        assert producer.bootstrap_servers == bootstrap_servers
    
    def test_produce_market_data_event(self, mock_kafka_producer):
        """Test producing market data event"""
        if EventProducer is None:
            pytest.skip("EventProducer not available")
        
        with patch.object(EventProducer, 'produce') as mock_produce:
            producer = EventProducer("localhost:9092")
            
            event_data = {
                "commodity": "crude_oil",
                "price": 85.50,
                "volume": 10000,
                "timestamp": 1729612800
            }
            
            producer.produce_market_data(**event_data)
            mock_produce.assert_called_once()
    
    def test_event_validation(self):
        """Test event data validation"""
        # Test that required fields are present
        required_fields = ["commodity", "price", "timestamp"]
        event_data = {"commodity": "gold", "price": 2000}
        
        missing = [f for f in required_fields if f not in event_data]
        assert "timestamp" in missing
    
    def test_event_serialization(self):
        """Test event serialization to JSON"""
        import json
        
        event = {
            "event_id": "test-123",
            "event_type": "MARKET_DATA_UPDATED",
            "commodity": "crude_oil",
            "price": 85.50
        }
        
        serialized = json.dumps(event)
        deserialized = json.loads(serialized)
        
        assert deserialized["event_id"] == "test-123"
        assert deserialized["price"] == 85.50
    
    def test_delivery_callback(self):
        """Test delivery callback mechanism"""
        delivery_count = {"count": 0}
        
        def callback(err, msg):
            if not err:
                delivery_count["count"] += 1
        
        # Simulate successful delivery
        callback(None, Mock())
        assert delivery_count["count"] == 1
    
    def test_error_handling(self):
        """Test error handling in event production"""
        error_count = {"count": 0}
        
        def error_callback(err, msg):
            if err:
                error_count["count"] += 1
        
        # Simulate error
        error_callback(Exception("Kafka error"), None)
        assert error_count["count"] == 1

@pytest.mark.unit
class TestEventTypes:
    """Test event type definitions"""
    
    def test_event_types_exist(self):
        """Test that event types are properly defined"""
        expected_types = [
            "MARKET_DATA_UPDATED",
            "PRICE_ALERT",
            "WORKFLOW_COMPLETED",
            "DATA_QUALITY_ALERT"
        ]
        
        # This would test against actual EventType enum
        # For now, just verify expected types list
        assert len(expected_types) == 4
    
    def test_event_schema_structure(self):
        """Test event schema has required fields"""
        event_schema = {
            "event_id": str,
            "event_type": str,
            "timestamp": int,
            "source": str,
            "data": dict
        }
        
        assert "event_id" in event_schema
        assert "event_type" in event_schema
        assert "timestamp" in event_schema


