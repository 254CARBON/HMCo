"""
Feed-run ledger service for idempotent feed ingestion.

Tracks watermarks, checksums, and state for each feed to enable:
- Exactly-once semantics
- Painless backfills
- Resume/replay capabilities
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import clickhouse_connect
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False


class FeedState(Enum):
    """Feed run states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLAYING = "replaying"


@dataclass
class FeedLedgerEntry:
    """Feed ledger entry representing a single feed run."""
    feed_id: str
    partition: str
    watermark_ts: datetime
    checksum: str
    state: FeedState
    run_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedLedger:
    """Feed ledger for tracking ingestion watermarks and state."""

    def __init__(self, backend: str = "postgres", connection_string: Optional[str] = None):
        """
        Initialize feed ledger.
        
        Args:
            backend: Storage backend ('postgres' or 'clickhouse')
            connection_string: Connection string for the backend
        """
        self.backend = backend
        self.connection_string = connection_string or self._get_default_connection_string()
        self._init_tables()

    def _get_default_connection_string(self) -> str:
        """Get default connection string from environment."""
        if self.backend == "postgres":
            return os.getenv(
                "FEED_LEDGER_POSTGRES_URL",
                "postgresql://postgres:postgres@postgres:5432/feed_ledger"
            )
        elif self.backend == "clickhouse":
            return os.getenv(
                "FEED_LEDGER_CLICKHOUSE_URL",
                "clickhouse://default:@clickhouse:9000/feed_ledger"
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _init_tables(self):
        """Initialize ledger tables."""
        if self.backend == "postgres":
            self._init_postgres_tables()
        elif self.backend == "clickhouse":
            self._init_clickhouse_tables()

    def _init_postgres_tables(self):
        """Initialize PostgreSQL tables."""
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available")
        
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feed_ledger (
                id SERIAL PRIMARY KEY,
                feed_id VARCHAR(255) NOT NULL,
                partition VARCHAR(255) NOT NULL,
                watermark_ts TIMESTAMP NOT NULL,
                checksum VARCHAR(64) NOT NULL,
                state VARCHAR(50) NOT NULL,
                run_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                UNIQUE(feed_id, partition, watermark_ts)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feed_ledger_feed_partition 
            ON feed_ledger(feed_id, partition)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feed_ledger_state 
            ON feed_ledger(state)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()

    def _init_clickhouse_tables(self):
        """Initialize ClickHouse tables."""
        if not CLICKHOUSE_AVAILABLE:
            raise ImportError("clickhouse-connect not available")
        
        client = clickhouse_connect.get_client(host=self.connection_string.split("@")[1].split(":")[0])
        
        client.command("""
            CREATE TABLE IF NOT EXISTS feed_ledger (
                id UInt64,
                feed_id String,
                partition String,
                watermark_ts DateTime,
                checksum String,
                state String,
                run_id Nullable(String),
                created_at DateTime DEFAULT now(),
                updated_at DateTime DEFAULT now(),
                metadata String
            ) ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY (feed_id, partition, watermark_ts)
        """)

    def write_entry(self, entry: FeedLedgerEntry) -> bool:
        """
        Write entry to feed ledger.
        
        Args:
            entry: Feed ledger entry
            
        Returns:
            True if successful
        """
        if self.backend == "postgres":
            return self._write_postgres_entry(entry)
        elif self.backend == "clickhouse":
            return self._write_clickhouse_entry(entry)
        return False

    def _write_postgres_entry(self, entry: FeedLedgerEntry) -> bool:
        """Write entry to PostgreSQL."""
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feed_ledger 
            (feed_id, partition, watermark_ts, checksum, state, run_id, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (feed_id, partition, watermark_ts)
            DO UPDATE SET 
                state = EXCLUDED.state,
                updated_at = CURRENT_TIMESTAMP,
                metadata = EXCLUDED.metadata
        """, (
            entry.feed_id,
            entry.partition,
            entry.watermark_ts,
            entry.checksum,
            entry.state.value,
            entry.run_id,
            json.dumps(entry.metadata) if entry.metadata else None
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True

    def _write_clickhouse_entry(self, entry: FeedLedgerEntry) -> bool:
        """Write entry to ClickHouse."""
        client = clickhouse_connect.get_client(host=self.connection_string.split("@")[1].split(":")[0])
        
        client.insert('feed_ledger', [[
            hash(f"{entry.feed_id}:{entry.partition}:{entry.watermark_ts}") % (2**64),
            entry.feed_id,
            entry.partition,
            entry.watermark_ts,
            entry.checksum,
            entry.state.value,
            entry.run_id,
            datetime.now(),
            datetime.now(),
            json.dumps(entry.metadata) if entry.metadata else "{}"
        ]])
        return True

    def get_latest_watermark(self, feed_id: str, partition: str) -> Optional[datetime]:
        """
        Get latest watermark for feed/partition.
        
        Args:
            feed_id: Feed identifier
            partition: Partition identifier
            
        Returns:
            Latest watermark timestamp or None
        """
        if self.backend == "postgres":
            return self._get_postgres_watermark(feed_id, partition)
        elif self.backend == "clickhouse":
            return self._get_clickhouse_watermark(feed_id, partition)
        return None

    def _get_postgres_watermark(self, feed_id: str, partition: str) -> Optional[datetime]:
        """Get latest watermark from PostgreSQL."""
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT watermark_ts FROM feed_ledger
            WHERE feed_id = %s AND partition = %s AND state = %s
            ORDER BY watermark_ts DESC
            LIMIT 1
        """, (feed_id, partition, FeedState.COMPLETED.value))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result[0] if result else None

    def _get_clickhouse_watermark(self, feed_id: str, partition: str) -> Optional[datetime]:
        """Get latest watermark from ClickHouse."""
        client = clickhouse_connect.get_client(host=self.connection_string.split("@")[1].split(":")[0])
        
        result = client.query("""
            SELECT watermark_ts FROM feed_ledger
            WHERE feed_id = {feed_id:String} AND partition = {partition:String} AND state = {state:String}
            ORDER BY watermark_ts DESC
            LIMIT 1
        """, parameters={
            'feed_id': feed_id,
            'partition': partition,
            'state': FeedState.COMPLETED.value
        })
        
        return result.result_rows[0][0] if result.result_rows else None

    def compute_checksum(self, data: bytes) -> str:
        """
        Compute SHA256 checksum of data.
        
        Args:
            data: Data bytes
            
        Returns:
            Hex checksum string
        """
        return hashlib.sha256(data).hexdigest()

    def list_entries(
        self,
        feed_id: Optional[str] = None,
        partition: Optional[str] = None,
        state: Optional[FeedState] = None,
        limit: int = 100
    ) -> List[FeedLedgerEntry]:
        """
        List feed ledger entries.
        
        Args:
            feed_id: Optional feed filter
            partition: Optional partition filter
            state: Optional state filter
            limit: Max results
            
        Returns:
            List of feed ledger entries
        """
        if self.backend == "postgres":
            return self._list_postgres_entries(feed_id, partition, state, limit)
        elif self.backend == "clickhouse":
            return self._list_clickhouse_entries(feed_id, partition, state, limit)
        return []

    def _list_postgres_entries(
        self,
        feed_id: Optional[str],
        partition: Optional[str],
        state: Optional[FeedState],
        limit: int
    ) -> List[FeedLedgerEntry]:
        """List entries from PostgreSQL."""
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM feed_ledger WHERE 1=1"
        params = []
        
        if feed_id:
            query += " AND feed_id = %s"
            params.append(feed_id)
        if partition:
            query += " AND partition = %s"
            params.append(partition)
        if state:
            query += " AND state = %s"
            params.append(state.value)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [
            FeedLedgerEntry(
                feed_id=row['feed_id'],
                partition=row['partition'],
                watermark_ts=row['watermark_ts'],
                checksum=row['checksum'],
                state=FeedState(row['state']),
                run_id=row.get('run_id'),
                created_at=row.get('created_at'),
                updated_at=row.get('updated_at'),
                metadata=json.loads(row['metadata']) if row.get('metadata') else None
            )
            for row in results
        ]

    def _list_clickhouse_entries(
        self,
        feed_id: Optional[str],
        partition: Optional[str],
        state: Optional[FeedState],
        limit: int
    ) -> List[FeedLedgerEntry]:
        """List entries from ClickHouse."""
        client = clickhouse_connect.get_client(host=self.connection_string.split("@")[1].split(":")[0])
        
        query = "SELECT * FROM feed_ledger WHERE 1=1"
        params = {}
        
        if feed_id:
            query += " AND feed_id = {feed_id:String}"
            params['feed_id'] = feed_id
        if partition:
            query += " AND partition = {partition:String}"
            params['partition'] = partition
        if state:
            query += " AND state = {state:String}"
            params['state'] = state.value
        
        query += " ORDER BY created_at DESC LIMIT {limit:UInt32}"
        params['limit'] = limit
        
        result = client.query(query, parameters=params)
        
        return [
            FeedLedgerEntry(
                feed_id=row[1],
                partition=row[2],
                watermark_ts=row[3],
                checksum=row[4],
                state=FeedState(row[5]),
                run_id=row[6],
                created_at=row[7],
                updated_at=row[8],
                metadata=json.loads(row[9]) if row[9] else None
            )
            for row in result.result_rows
        ]
