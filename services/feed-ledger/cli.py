"""
CLI for feed-ledger replay and management.

Usage:
  feeds replay --feed iso.rt.lmp.caiso --from 2025-07-01T00:00Z --to now
  feeds list --feed iso.rt.lmp.caiso --state completed
  feeds status --feed iso.rt.lmp.caiso
"""

import sys
import argparse
from datetime import datetime, timezone
from typing import Optional
import logging

from feed_ledger import FeedLedger, FeedState, FeedLedgerEntry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string."""
    if dt_str.lower() == "now":
        return datetime.now(timezone.utc)
    
    # Try ISO format
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except ValueError:
        pass
    
    # Try common formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse datetime: {dt_str}")


def cmd_replay(args):
    """Execute replay command."""
    ledger = FeedLedger(backend=args.backend)
    
    from_ts = parse_datetime(args.from_time)
    to_ts = parse_datetime(args.to_time)
    
    logger.info(f"Replaying feed '{args.feed}' from {from_ts} to {to_ts}")
    
    # Get entries in time range
    entries = ledger.list_entries(
        feed_id=args.feed,
        partition=args.partition,
        limit=10000
    )
    
    # Filter by time range
    replay_entries = [
        e for e in entries
        if from_ts <= e.watermark_ts <= to_ts
    ]
    
    logger.info(f"Found {len(replay_entries)} entries to replay")
    
    # Mark entries for replay
    for entry in replay_entries:
        entry.state = FeedState.REPLAYING
        ledger.write_entry(entry)
        logger.info(f"Marked for replay: {entry.feed_id} partition={entry.partition} watermark={entry.watermark_ts}")
    
    logger.info(f"Replay initiated for {len(replay_entries)} entries")
    logger.info("Use DolphinScheduler to trigger the corresponding workflows")


def cmd_list(args):
    """Execute list command."""
    ledger = FeedLedger(backend=args.backend)
    
    state = FeedState(args.state) if args.state else None
    
    entries = ledger.list_entries(
        feed_id=args.feed,
        partition=args.partition,
        state=state,
        limit=args.limit
    )
    
    logger.info(f"Found {len(entries)} entries")
    
    for entry in entries:
        print(f"Feed: {entry.feed_id}")
        print(f"  Partition: {entry.partition}")
        print(f"  Watermark: {entry.watermark_ts}")
        print(f"  State: {entry.state.value}")
        print(f"  Checksum: {entry.checksum[:16]}...")
        if entry.run_id:
            print(f"  Run ID: {entry.run_id}")
        print(f"  Updated: {entry.updated_at}")
        print()


def cmd_status(args):
    """Execute status command."""
    ledger = FeedLedger(backend=args.backend)
    
    entries = ledger.list_entries(feed_id=args.feed, limit=1000)
    
    if not entries:
        logger.info(f"No entries found for feed '{args.feed}'")
        return
    
    # Compute statistics
    state_counts = {}
    latest_watermark = None
    earliest_watermark = None
    
    for entry in entries:
        state_counts[entry.state.value] = state_counts.get(entry.state.value, 0) + 1
        
        if latest_watermark is None or entry.watermark_ts > latest_watermark:
            latest_watermark = entry.watermark_ts
        
        if earliest_watermark is None or entry.watermark_ts < earliest_watermark:
            earliest_watermark = entry.watermark_ts
    
    print(f"Status for feed: {args.feed}")
    print(f"  Total entries: {len(entries)}")
    print(f"  Latest watermark: {latest_watermark}")
    print(f"  Earliest watermark: {earliest_watermark}")
    print(f"  State breakdown:")
    for state, count in sorted(state_counts.items()):
        print(f"    {state}: {count}")


def cmd_watermark(args):
    """Get latest watermark for feed/partition."""
    ledger = FeedLedger(backend=args.backend)
    
    watermark = ledger.get_latest_watermark(args.feed, args.partition)
    
    if watermark:
        print(f"Latest watermark for {args.feed}/{args.partition}: {watermark}")
    else:
        print(f"No watermark found for {args.feed}/{args.partition}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Feed ledger CLI')
    parser.add_argument('--backend', choices=['postgres', 'clickhouse'], default='postgres',
                       help='Storage backend')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Replay feed data')
    replay_parser.add_argument('--feed', required=True, help='Feed ID')
    replay_parser.add_argument('--partition', help='Partition (optional)')
    replay_parser.add_argument('--from', dest='from_time', required=True,
                              help='Start time (ISO format or "now")')
    replay_parser.add_argument('--to', dest='to_time', required=True,
                              help='End time (ISO format or "now")')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List feed entries')
    list_parser.add_argument('--feed', help='Feed ID filter')
    list_parser.add_argument('--partition', help='Partition filter')
    list_parser.add_argument('--state', help='State filter')
    list_parser.add_argument('--limit', type=int, default=100, help='Max results')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show feed status')
    status_parser.add_argument('--feed', required=True, help='Feed ID')
    
    # Watermark command
    watermark_parser = subparsers.add_parser('watermark', help='Get latest watermark')
    watermark_parser.add_argument('--feed', required=True, help='Feed ID')
    watermark_parser.add_argument('--partition', required=True, help='Partition')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'replay':
        cmd_replay(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'watermark':
        cmd_watermark(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
