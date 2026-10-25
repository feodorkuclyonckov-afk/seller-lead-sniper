"""Sprint 2 — local message deduplication backed by SQLite.

Purpose
-------
Telegram feeds contain a large amount of cross-posted content: the same "ищу
карго" request is often forwarded into a dozen chats, and channels repost each
other verbatim. Sending every copy to the (paid) LLM classifier wastes money
and pollutes the lead table.

This module keeps a lightweight, local cache of every message body already
processed by the pipeline. Lookups are cheap and, crucially, non-blocking:
``aiosqlite`` moves the blocking ``sqlite3`` work onto a worker thread so the
``asyncio`` event loop in charge of the Telethon client never stalls.

Public API
----------
``init_db(db_path)``
    Create the ``seen_messages`` table (idempotent).
``hash_text(text)``
    Normalise a message body and return its MD5 hex digest.
``is_duplicate(text, db_path)``
    Atomically check-and-record a message. Returns ``True`` when the message
    was seen before, ``False`` when it is new (and records it).

Notes
-----
* The hash is intentionally computed over a *normalised* body (case-folded,
  punctuation stripped, whitespace collapsed) so that cosmetic differences
  between reposts do not defeat the cache.
* ``is_duplicate`` uses a single ``INSERT OR IGNORE`` statement, which makes
  the check-and-insert atomic across concurrent scrapers sharing one DB file.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Final

import aiosqlite

LOGGER = logging.getLogger("lead_sniper.db")

#: Default on-disk location of the deduplication cache.
DEFAULT_DB_PATH: Final[str] = "leads_cache.db"

#: How long SQLite waits for a competing writer before raising "database is
#: locked", in seconds.
BUSY_TIMEOUT: Final[float] = 5.0

_CREATE_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS seen_messages (
    msg_hash TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT_OR_IGNORE_SQL: Final[str] = (
    "INSERT OR IGNORE INTO seen_messages (msg_hash) VALUES (?)"
)

# Replace every run of punctuation/symbols (and underscore) with a single
# space. ``\W`` is Unicode-aware, so Cyrillic letters survive untouched.
_NON_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[\W_]+", re.UNICODE)


def _prepare_path(db_path: str | Path) -> Path:
    """Return the database path as a :class:`Path` and ensure its parent exists."""
    path = Path(db_path)
    parent = path.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    return path


def hash_text(text: str) -> str:
    """Return the MD5 hex digest of a normalised message body.

    Normalisation performs three steps, in order:

    1. Unicode ``NFKC`` normalisation (folds visually identical characters
       such as full-width or composed forms).
    2. Case folding via ``str.lower()``.
    3. Collapsing every run of punctuation/symbols and whitespace into a
       single space, then trimming the ends.

    The result is a 32-character lowercase hexadecimal string.

    Raises:
        TypeError: if ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"hash_text() expects str, got {type(text).__name__}"
        )

    normalised = unicodedata.normalize("NFKC", text).lower()
    normalised = _NON_WORD_RE.sub(" ", normalised).strip()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


async def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the ``seen_messages`` table if it does not exist yet.

    Safe to call on every process start. WAL journaling is enabled to allow
    readers and a single writer to coexist without blocking each other.

    Args:
        db_path: Path to the SQLite database file.
    """
    path = _prepare_path(db_path)
    async with aiosqlite.connect(path, timeout=BUSY_TIMEOUT) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute(_CREATE_TABLE_SQL)
        await conn.commit()
    LOGGER.debug("Deduplication cache ready at %s", path)


async def is_duplicate(text: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Check ``text`` against the cache, recording it when it is new.

    The check-and-insert is a single ``INSERT OR IGNORE`` statement: if the
    normalised hash was already present, SQLite ignores the insert and the
    function reports ``True``. Otherwise the row is written and ``False`` is
    returned, so the same call both *tests* and *reserves* a message.

    Args:
        text: The raw message body.
        db_path: Path to the SQLite database file.

    Returns:
        ``True`` if the message (after normalisation) was seen before,
        ``False`` if it is new.

    Raises:
        TypeError: if ``text`` is not a ``str``.
    """
    digest = hash_text(text)
    path = _prepare_path(db_path)

    async with aiosqlite.connect(path, timeout=BUSY_TIMEOUT) as conn:
        try:
            cursor = await conn.execute(_INSERT_OR_IGNORE_SQL, (digest,))
        except sqlite3.OperationalError:
            # The cache was opened before ``init_db()`` ran (e.g. a fresh
            # worker). Create the schema once and retry immediately.
            await conn.execute(_CREATE_TABLE_SQL)
            cursor = await conn.execute(_INSERT_OR_IGNORE_SQL, (digest,))

        inserted = cursor.rowcount == 1
        await cursor.close()
        await conn.commit()

    if inserted:
        LOGGER.debug("New message recorded (hash=%s)", digest)
    else:
        LOGGER.debug("Duplicate message skipped (hash=%s)", digest)
    return not inserted
