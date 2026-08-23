"""Engine and session construction.

SQLite and PostgreSQL are both first-class here, and the difference is
contained to this file. SQLite is what makes ``pip install`` the entire setup
story -- no container, no server, clone and run. PostgreSQL is what a
deployment would use, and CI runs the whole suite against a real one so the
SQLite convenience never becomes a lie about where the code has been tested.

The two engine configurations differ in ways that are easy to get wrong:

* SQLite needs ``check_same_thread=False`` because FastAPI's threadpool runs
  sync endpoints on worker threads, and needs *no* pool sizing arguments --
  passing them to the default SQLite pool raises.
* PostgreSQL gets an explicit pool and ``pool_pre_ping``, which turns a
  connection killed by a network device into one retried checkout instead of
  one failed request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Base


def create_db_engine(settings: Settings) -> Engine:
    """Build the engine for the configured database URL."""
    kwargs: dict[str, Any] = {"future": True, "echo": False}

    if settings.is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.db_pool_size
        kwargs["max_overflow"] = settings.db_pool_max_overflow
        kwargs["pool_pre_ping"] = True

    engine = create_engine(settings.database_url, **kwargs)

    if settings.is_sqlite:
        _enable_sqlite_foreign_keys(engine)

    return engine


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Turn on foreign key enforcement.

    SQLite ignores foreign keys unless asked, per connection. Without this the
    ``ON DELETE CASCADE`` declarations are decoration on SQLite and real on
    PostgreSQL -- the two backends would behave differently on delete, and the
    local one would be the one that silently leaves orphans.
    """

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory with autoflush off.

    Autoflush is disabled deliberately: the workflow builds a run's records
    incrementally and reads its own in-progress state, and implicit flushes
    would push partial rows at unpredictable points. Commits are explicit.
    """
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create the schema.

    Sufficient because this service's schema is created once and never
    migrated in place -- there is no production data to preserve. A service
    that did would want Alembic here instead, and saying so is more honest
    than shipping a migrations directory with one revision in it.
    """
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transaction boundary: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
