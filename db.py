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
    Create the ``seen_messages`` and ``discovered_chats`` tables (idempotent).
``hash_text(text)``
    Normalise a message body and return its MD5 hex digest.
``is_duplicate(text, db_path)``
    Atomically check-and-record a message. Returns ``True`` when the message
    was seen before, ``False`` when it is new (and records it).

Discovery audit trail (Sprint 5, :mod:`auto_discover`)
``is_chat_evaluated(chat_id, db_path)``
    ``True`` when the chat already has an audit row in ``discovered_chats``.
``get_chat_status(chat_id, db_path)``
    Last recorded status of a chat, or ``None`` when it was never audited.
``record_chat_evaluation(...)``
    Upsert the audit row (username, title, members, last message, status) and
    stamp ``joined_at`` when the chat was actually joined.
``get_daily_joined_count(db_path)``
    Number of chats joined during the current UTC day, used to enforce
    ``DISCOVERY_DAILY_JOIN_LIMIT``.

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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Optional

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

# --------------------------------------------------------------------------- #
# Discovery audit trail (Sprint 5 — auto_discover.py)
# --------------------------------------------------------------------------- #

#: Audit table for every public chat the discovery pipeline has looked at.
#: ``status`` is a free-form TEXT column; the recognised values are the
#: :data:`STATUS_*` constants below.
_CREATE_DISCOVERED_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS discovered_chats (
    chat_id INTEGER PRIMARY KEY,
    username TEXT,
    title TEXT,
    members_count INTEGER,
    last_message_date TIMESTAMP,
    status TEXT,
    discovered_at TIMESTAMP,
    joined_at TIMESTAMP NULL
)
"""

# Upsert: re-evaluating a chat refreshes the audit fields but never moves
# ``discovered_at`` backwards and never clears a previously recorded join.
_UPSERT_DISCOVERED_SQL: Final[str] = """
INSERT INTO discovered_chats (
    chat_id, username, title, members_count, last_message_date,
    status, discovered_at, joined_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id) DO UPDATE SET
    username=excluded.username,
    title=excluded.title,
    members_count=excluded.members_count,
    last_message_date=excluded.last_message_date,
    status=excluded.status,
    joined_at=COALESCE(excluded.joined_at, discovered_chats.joined_at)
"""

_SELECT_EVALUATED_SQL: Final[str] = (
    "SELECT 1 FROM discovered_chats WHERE chat_id = ? LIMIT 1"
)

_SELECT_STATUS_SQL: Final[str] = (
    "SELECT status FROM discovered_chats WHERE chat_id = ? LIMIT 1"
)

# ``date('now')`` is UTC and SQLite normalises the ``+00:00`` offset stored by
# ``record_chat_evaluation``, so the comparison is timezone-correct.
_COUNT_JOINED_TODAY_SQL: Final[str] = (
    "SELECT COUNT(*) FROM discovered_chats "
    "WHERE joined_at IS NOT NULL AND date(joined_at) = date('now')"
)

#: Chat joined during this (or a previous) discovery session.
STATUS_JOINED: Final[str] = "joined"
#: Chat exists but shows no recent activity (stale / abandoned).
STATUS_INACTIVE: Final[str] = "inactive"
#: Chat reports fewer members than ``DISCOVERY_MIN_MEMBERS``.
STATUS_TOO_FEW_MEMBERS: Final[str] = "too_few_members"
#: The account is already a member; recorded so the chat is never re-tried.
STATUS_ALREADY_JOINED: Final[str] = "already_joined"
#: Structurally unusable chat (broadcast-only, deactivated, no username).
STATUS_BLACKLISTED: Final[str] = "blacklisted"
#: Passed every pre-join gate but was not actually joined (dry-run preview).
STATUS_ELIGIBLE: Final[str] = "eligible"

#: Every value :func:`record_chat_evaluation` accepts.
KNOWN_STATUSES: Final[frozenset[str]] = frozenset(
    {
        STATUS_JOINED,
        STATUS_INACTIVE,
        STATUS_TOO_FEW_MEMBERS,
        STATUS_ALREADY_JOINED,
        STATUS_BLACKLISTED,
        STATUS_ELIGIBLE,
    }
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
        await conn.execute(_CREATE_DISCOVERED_TABLE_SQL)
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


# --------------------------------------------------------------------------- #
# Discovery audit helpers
# --------------------------------------------------------------------------- #


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC :class:`datetime`."""
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    """Return ``now`` as an ISO-8601 string carrying an explicit UTC offset."""
    return _utc_now().isoformat()


def _coerce_timestamp(value: Optional[datetime | str]) -> Optional[str]:
    """Normalise a timestamp argument to an ISO-8601 UTC string (or ``None``).

    Naive ``datetime`` values are assumed to be UTC: every Telegram timestamp
    is timezone-aware, so a naive value only appears in hand-made callers.

    Raises:
        TypeError: if ``value`` is neither ``None``, a ``datetime`` nor a
            ``str``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        return value.strip() or None
    raise TypeError(
        "last_message_date must be datetime, str or None, "
        f"got {type(value).__name__}"
    )


@asynccontextmanager
async def _open_audit_db(path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """Open ``path`` and make sure the discovery audit table exists.

    Yields the connection inside an ``async with`` block: awaiting the
    connection *and* entering its async context manager would start the
    aiosqlite worker thread twice (``RuntimeError: threads can only be
    started once``), so the context manager is created exactly once here.
    """
    async with aiosqlite.connect(path, timeout=BUSY_TIMEOUT) as conn:
        await conn.execute(_CREATE_DISCOVERED_TABLE_SQL)
        yield conn


async def is_chat_evaluated(
    chat_id: int, db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Return ``True`` when ``chat_id`` already has an audit row.

    Args:
        chat_id: Numeric Telegram id of the chat.
        db_path: Path to the SQLite database file.

    Raises:
        TypeError: if ``chat_id`` is not an ``int``.
    """
    if not isinstance(chat_id, int) or isinstance(chat_id, bool):
        raise TypeError(
            f"is_chat_evaluated() expects int, got {type(chat_id).__name__}"
        )
    path = _prepare_path(db_path)
    async with _open_audit_db(path) as conn:
        cursor = await conn.execute(_SELECT_EVALUATED_SQL, (chat_id,))
        row = await cursor.fetchone()
        await cursor.close()
    return row is not None


async def get_chat_status(
    chat_id: int, db_path: str = DEFAULT_DB_PATH
) -> Optional[str]:
    """Return the last recorded status for ``chat_id``, or ``None``.

    Args:
        chat_id: Numeric Telegram id of the chat.
        db_path: Path to the SQLite database file.

    Raises:
        TypeError: if ``chat_id`` is not an ``int``.
    """
    if not isinstance(chat_id, int) or isinstance(chat_id, bool):
        raise TypeError(
            f"get_chat_status() expects int, got {type(chat_id).__name__}"
        )
    path = _prepare_path(db_path)
    async with _open_audit_db(path) as conn:
        cursor = await conn.execute(_SELECT_STATUS_SQL, (chat_id,))
        row = await cursor.fetchone()
        await cursor.close()
    return str(row[0]) if row and row[0] is not None else None


async def record_chat_evaluation(
    chat_id: int,
    username: Optional[str] = None,
    title: Optional[str] = None,
    members_count: Optional[int] = None,
    last_message_date: Optional[datetime | str] = None,
    status: str = STATUS_INACTIVE,
    joined: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Insert or update the audit row for ``chat_id``.

    The write is an upsert: ``discovered_at`` keeps the *first* observation,
    ``joined_at`` is only set when ``joined`` is true (and is never cleared by
    a later non-join evaluation), and every other audit field is refreshed.

    Args:
        chat_id: Numeric Telegram id of the chat (primary key).
        username: Public username without ``@``, when known.
        title: Human-readable chat title.
        members_count: ``participants_count`` reported by Telegram.
        last_message_date: Timestamp of the newest observed message. Naive
            datetimes are treated as UTC.
        status: One of :data:`KNOWN_STATUSES`.
        joined: Set to ``True`` when this evaluation records an actual join.
        db_path: Path to the SQLite database file.

    Raises:
        TypeError: if ``chat_id`` is not an ``int``.
        ValueError: if ``status`` is not a recognised value.
    """
    if not isinstance(chat_id, int) or isinstance(chat_id, bool):
        raise TypeError(
            "record_chat_evaluation() expects int chat_id, "
            f"got {type(chat_id).__name__}"
        )
    if status not in KNOWN_STATUSES:
        raise ValueError(
            f"Unknown chat status {status!r}; expected one of "
            f"{sorted(KNOWN_STATUSES)}"
        )

    now_iso = _utcnow_iso()
    joined_at = now_iso if joined else None
    last_seen = _coerce_timestamp(last_message_date)
    path = _prepare_path(db_path)

    async with _open_audit_db(path) as conn:
        await conn.execute(
            _UPSERT_DISCOVERED_SQL,
            (
                chat_id,
                username,
                title,
                members_count,
                last_seen,
                status,
                now_iso,
                joined_at,
            ),
        )
        await conn.commit()

    LOGGER.debug(
        "Chat %s (%s) recorded with status=%s (joined=%s).",
        chat_id,
        username or "no_username",
        status,
        joined,
    )


async def get_daily_joined_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """Return the number of chats joined during the current UTC day.

    Counts every audit row whose ``joined_at`` falls on today's UTC date,
    regardless of the status recorded afterwards.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The number of joins recorded today (``0`` when the table is empty).
    """
    path = _prepare_path(db_path)
    async with _open_audit_db(path) as conn:
        cursor = await conn.execute(_COUNT_JOINED_TODAY_SQL)
        row = await cursor.fetchone()
        await cursor.close()
    return int(row[0]) if row and row[0] is not None else 0
