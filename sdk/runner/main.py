#!/usr/bin/env python3
"""
Main entry point for the HMCo Data Platform Ingestion Runner.

This application executes data ingestion jobs based on Unified Ingestion Spec (UIS)
configurations using various execution engines (SeaTunnel, Spark, Flink).
"""

import argparse
import json
import sys
import logging
import signal
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the SDK path to Python path
sdk_path = Path(__file__).parent.parent
sys.path.insert(0, str(sdk_path))

from runner import IngestionRunner, RunnerConfig
from runner.config import LogLevel


def setup_signal_handlers(runner: IngestionRunner):
    """Setup signal handlers for graceful shutdown."""
    logger = logging.getLogger(__name__)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HMCo Data Platform Ingestion Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sample job
  python main.py sample examples/polygon-api.yaml

  # Start HTTP server
  python main.py server --host 0.0.0.0 --port 8080

  # Run with custom configuration
  python main.py sample spec.yaml --config custom-config.json

  # List available compilers
  python main.py info
        """
    )

    parser.add_argument("command", choices=["sample", "server", "info", "health"],
                       help="Command to execute")

    # Common options
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--log-level", choices=[l.value for l in LogLevel],
                       default="info", help="Log level")
    parser.add_argument("--vault-token", help="Vault authentication token")
    parser.add_argument("--tenant-id", help="Tenant ID")

    # Sample command options
    parser.add_argument("spec_path", nargs="?", help="Path to UIS specification file")

    # Server command options
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    args = parser.parse_args()

    # Setup logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = RunnerConfig.from_dict(config_dict)
        else:
            config = RunnerConfig.from_env()

        # Override with command line arguments
        if args.vault_token:
            config.vault_token = args.vault_token

        if args.tenant_id:
            config.tenant_id = args.tenant_id

        # Create runner
        runner = IngestionRunner(config)

        # Setup signal handlers
        setup_signal_handlers(runner)

        # Execute command
        if args.command == "sample":
            if not args.spec_path:
                parser.error("spec_path required for sample command")

            logger.info(f"Running sample job: {args.spec_path}")

            result = runner.run_sample_job(args.spec_path)

            if result["success"]:
                logger.info("Sample job completed successfully")
                print(f"Job result: {result}")
                return 0
            else:
                logger.error(f"Sample job failed: {result.get('error', 'Unknown error')}")
                return 1

        elif args.command == "server":
            logger.info(f"Starting HTTP server on {args.host}:{args.port}")

            # Check if FastAPI is available
            try:
                from fastapi import FastAPI
                runner.run_http_server(host=args.host, port=args.port)
                return 0
            except ImportError:
                logger.error("FastAPI not available, cannot start HTTP server")
                logger.error("Install with: pip install fastapi uvicorn")
                return 1

        elif args.command == "info":
            print("HMCo Data Platform Ingestion Runner")
            print("=" * 40)
            print(f"Runner ID: {config.runner_id}")
            print(f"Tenant ID: {config.tenant_id}")
            print(f"Max concurrent jobs: {config.max_concurrent_jobs}")
            print(f"Job timeout: {config.job_timeout_seconds}s")
            print()
            print("Available engines:")
            for engine in config.supported_engines:
                status = "✓" if engine in runner.compilers else "✗"
                print(f"  {status} {engine.value}")

            if runner.secret_manager:
                vault_status = "✓" if runner.secret_manager.is_enabled() else "✗"
                print(f"  {vault_status} Vault integration")

            metrics_status = "✓" if config.metrics_enabled else "✗"
            print(f"  {metrics_status} Metrics collection")

            tracing_status = "✓" if config.tracing_enabled else "✗"
            print(f"  {tracing_status} Distributed tracing")

            return 0

        elif args.command == "health":
            health = runner.get_health_status()
            print(json.dumps(health, indent=2))

            if health["status"] != "healthy":
                return 1
            return 0

    except Exception as e:
        logger.error(f"Runner failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
