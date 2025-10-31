"""
Tests for feed ledger service.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feed_ledger import FeedLedger, FeedLedgerEntry, FeedState


class TestFeedLedgerEntry:
    """Test FeedLedgerEntry dataclass."""

    def test_create_entry(self):
        """Test creating a feed ledger entry."""
        entry = FeedLedgerEntry(
            feed_id="iso.rt.lmp.caiso",
            partition="2025-01-15",
            watermark_ts=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            checksum="abc123",
            state=FeedState.COMPLETED
        )
        
        assert entry.feed_id == "iso.rt.lmp.caiso"
        assert entry.partition == "2025-01-15"
        assert entry.state == FeedState.COMPLETED
        assert entry.checksum == "abc123"

    def test_entry_with_metadata(self):
        """Test entry with metadata."""
        entry = FeedLedgerEntry(
            feed_id="iso.rt.lmp.caiso",
            partition="2025-01-15",
            watermark_ts=datetime.now(timezone.utc),
            checksum="abc123",
            state=FeedState.RUNNING,
            run_id="run-123",
            metadata={"records": 1000, "source": "api"}
        )
        
        assert entry.run_id == "run-123"
        assert entry.metadata["records"] == 1000
        assert entry.metadata["source"] == "api"


class TestFeedState:
    """Test FeedState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert FeedState.PENDING.value == "pending"
        assert FeedState.RUNNING.value == "running"
        assert FeedState.COMPLETED.value == "completed"
        assert FeedState.FAILED.value == "failed"
        assert FeedState.REPLAYING.value == "replaying"


class TestFeedLedgerChecksum:
    """Test checksum computation."""

    def test_compute_checksum(self):
        """Test checksum computation."""
        # Mock the FeedLedger to avoid DB connection
        with patch.object(FeedLedger, '_init_tables', return_value=None):
            ledger = FeedLedger(backend="postgres")
            
            data = b"test data"
            checksum = ledger.compute_checksum(data)
            
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA256 hex digest

    def test_same_data_same_checksum(self):
        """Test same data produces same checksum."""
        with patch.object(FeedLedger, '_init_tables', return_value=None):
            ledger = FeedLedger(backend="postgres")
            
            data = b"test data"
            checksum1 = ledger.compute_checksum(data)
            checksum2 = ledger.compute_checksum(data)
            
            assert checksum1 == checksum2

    def test_different_data_different_checksum(self):
        """Test different data produces different checksum."""
        with patch.object(FeedLedger, '_init_tables', return_value=None):
            ledger = FeedLedger(backend="postgres")
            
            checksum1 = ledger.compute_checksum(b"data1")
            checksum2 = ledger.compute_checksum(b"data2")
            
            assert checksum1 != checksum2


class TestFeedLedgerConfiguration:
    """Test feed ledger configuration."""

    def test_default_postgres_backend(self):
        """Test default PostgreSQL backend."""
        with patch.object(FeedLedger, '_init_tables', return_value=None):
            ledger = FeedLedger(backend="postgres")
            assert ledger.backend == "postgres"

    def test_clickhouse_backend(self):
        """Test ClickHouse backend."""
        with patch.object(FeedLedger, '_init_tables', return_value=None):
            ledger = FeedLedger(backend="clickhouse")
            assert ledger.backend == "clickhouse"

    def test_custom_connection_string(self):
        """Test custom connection string."""
        with patch.object(FeedLedger, '_init_tables', return_value=None):
            conn_str = "postgresql://user:pass@host:5432/db"
            ledger = FeedLedger(backend="postgres", connection_string=conn_str)
            assert ledger.connection_string == conn_str

    def test_invalid_backend_raises_error(self):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError):
            with patch.object(FeedLedger, '_init_tables', return_value=None):
                ledger = FeedLedger(backend="invalid")
                ledger._get_default_connection_string()


@pytest.fixture
def mock_postgres_connection():
    """Mock PostgreSQL connection."""
    with patch('feed_ledger.psycopg2.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn, mock_cursor


@pytest.fixture
def mock_clickhouse_client():
    """Mock ClickHouse client."""
    with patch('feed_ledger.clickhouse_connect.get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


class TestFeedLedgerWriteEntry:
    """Test writing entries to feed ledger."""

    def test_write_postgres_entry(self, mock_postgres_connection):
        """Test writing entry to PostgreSQL."""
        with patch.object(FeedLedger, 'POSTGRES_AVAILABLE', True):
            ledger = FeedLedger(backend="postgres")
            
            entry = FeedLedgerEntry(
                feed_id="test.feed",
                partition="2025-01-15",
                watermark_ts=datetime.now(timezone.utc),
                checksum="abc123",
                state=FeedState.COMPLETED
            )
            
            result = ledger.write_entry(entry)
            assert result == True

    def test_write_clickhouse_entry(self, mock_clickhouse_client):
        """Test writing entry to ClickHouse."""
        with patch.object(FeedLedger, 'CLICKHOUSE_AVAILABLE', True):
            ledger = FeedLedger(backend="clickhouse")
            
            entry = FeedLedgerEntry(
                feed_id="test.feed",
                partition="2025-01-15",
                watermark_ts=datetime.now(timezone.utc),
                checksum="abc123",
                state=FeedState.COMPLETED
            )
            
            result = ledger.write_entry(entry)
            assert result == True


class TestFeedLedgerGetWatermark:
    """Test getting watermarks from feed ledger."""

    def test_get_watermark_postgres(self, mock_postgres_connection):
        """Test getting watermark from PostgreSQL."""
        mock_conn, mock_cursor = mock_postgres_connection
        
        # Mock return value
        expected_watermark = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_cursor.fetchone.return_value = (expected_watermark,)
        
        with patch.object(FeedLedger, 'POSTGRES_AVAILABLE', True):
            ledger = FeedLedger(backend="postgres")
            watermark = ledger.get_latest_watermark("test.feed", "2025-01-15")
            
            assert watermark == expected_watermark

    def test_get_watermark_not_found(self, mock_postgres_connection):
        """Test getting watermark when none exists."""
        mock_conn, mock_cursor = mock_postgres_connection
        mock_cursor.fetchone.return_value = None
        
        with patch.object(FeedLedger, 'POSTGRES_AVAILABLE', True):
            ledger = FeedLedger(backend="postgres")
            watermark = ledger.get_latest_watermark("test.feed", "2025-01-15")
            
            assert watermark is None


class TestFeedLedgerListEntries:
    """Test listing entries from feed ledger."""

    def test_list_all_entries(self, mock_postgres_connection):
        """Test listing all entries."""
        mock_conn, mock_cursor = mock_postgres_connection
        
        # Mock return values
        mock_cursor.fetchall.return_value = [
            {
                'feed_id': 'test.feed',
                'partition': '2025-01-15',
                'watermark_ts': datetime.now(timezone.utc),
                'checksum': 'abc123',
                'state': 'completed',
                'run_id': 'run-1',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'metadata': None
            }
        ]
        
        with patch.object(FeedLedger, 'POSTGRES_AVAILABLE', True):
            with patch('feed_ledger.RealDictCursor'):
                ledger = FeedLedger(backend="postgres")
                entries = ledger.list_entries(feed_id="test.feed", limit=100)
                
                assert len(entries) == 1
                assert entries[0].feed_id == "test.feed"
                assert entries[0].state == FeedState.COMPLETED

    def test_list_entries_filtered_by_state(self, mock_postgres_connection):
        """Test listing entries filtered by state."""
        mock_conn, mock_cursor = mock_postgres_connection
        mock_cursor.fetchall.return_value = []
        
        with patch.object(FeedLedger, 'POSTGRES_AVAILABLE', True):
            with patch('feed_ledger.RealDictCursor'):
                ledger = FeedLedger(backend="postgres")
                entries = ledger.list_entries(
                    feed_id="test.feed",
                    state=FeedState.FAILED,
                    limit=100
                )
                
                assert len(entries) == 0
