#!/usr/bin/env python3
"""
Database setup script for the portal backend.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from models.database import create_tables, drop_tables

def setup_database():
    """Set up the database tables."""
    print("Setting up database...")

    # Drop existing tables (for development)
    if os.getenv('DROP_EXISTING', 'false').lower() == 'true':
        print("Dropping existing tables...")
        drop_tables()

    # Create tables
    print("Creating tables...")
    create_tables()

    print("Database setup complete!")

if __name__ == "__main__":
    setup_database()
