"""
Database configuration and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, StaticPool

from . import Base

# Database configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://user:password@localhost:5432/portal'
)

def _resolve_pool_class(database_url: str):
    """Select an appropriate SQLAlchemy pool class for the configured database."""
    if database_url.startswith("sqlite"):
        # SQLite in-memory databases require StaticPool to reuse the same connection.
        return StaticPool
    return NullPool


# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    poolclass=_resolve_pool_class(DATABASE_URL),
    pool_pre_ping=True,
    future=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Create a new database session.

    Caller is responsible for closing the session via `session.close()`.
    """
    return SessionLocal()


def create_tables():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all tables."""
    Base.metadata.drop_all(bind=engine)


