"""Sprint 5 — smart public-chat discovery, pre-join auditing and a safe joiner.

Purpose
-------
Grow the monitored chat pool automatically without ever "join-spamming": the
module searches Telegram's public directory by the keywords listed in
``keywords.txt``, *evaluates* each candidate **before** issuing
``JoinChannelRequest`` and writes the full audit trail to SQLite
(:mod:`db`, table ``discovered_chats``).

Pre-join gates (in evaluation order)
------------------------------------
1. **Entity gate** — the chat must be a supergroup/megagroup (or a broadcast
   channel with a discussion group). Deactivated chats, chats without a public
   username and chats the account is already a member of are rejected.
2. **Member gate** — ``participants_count >= DISCOVERY_MIN_MEMBERS`` (50).
   A minimal entity that carries no count is expanded through
   ``GetFullChannelRequest`` (or ``GetFullChatRequest`` for basic groups)
   before the gate is applied, so active public chats are not rejected as
   "unknown".
3. **Freshness gate** — the newest of the last 10 messages must be no older
   than 48 hours, otherwise the chat is considered abandoned.
4. **Velocity & sender gate** — at least 5 messages in the last 7 days, posted
   by at least two distinct senders (a single bot pumping messages does not
   count as activity).

Anti-flood safeguards
---------------------
* ``DISCOVERY_DAILY_JOIN_LIMIT`` (default 5) joins per UTC day, checked against
  the SQLite audit table; reaching it exits gracefully with a warning.
* 10-25 s random jitter between search queries.
* 180-360 s random jitter between ``JoinChannelRequest`` calls.
* ``FloodWaitError`` is logged (with the reported seconds) and stops the run;
  ``UserAlreadyParticipantError`` is recorded as ``already_joined``;
  ``ChannelsTooMuchError`` stops the run and logs the account limit.

CLI
---
``python auto_discover.py --dry-run``
    Search + evaluate every candidate, print/log the audit line (title,
    members, last message date, pass/fail reason) and persist it, but never
    join.
``python auto_discover.py --limit 2``
    Join at most 2 chats in this run, additionally capped by the daily limit.
``--keyword TEXT``
    Override the keyword list (repeatable); ``keywords.txt`` is used by default.
``--db PATH``
    Audit SQLite file (default ``leads_cache.db``).

Exit codes mirror :mod:`join_chats`: ``0`` ok, ``1`` config problem, ``2``
session not authorized, ``3`` nothing joined, ``4`` unexpected failure.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import logging
import os
import random
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import httpx
from telethon import TelegramClient
from telethon.errors import (
    ChannelsTooMuchError,
    FloodWaitError,
    UserAlreadyParticipantError,
)
from telethon.tl.functions.channels import (
    GetFullChannelRequest,
    JoinChannelRequest,
)
from telethon.tl.functions.contacts import SearchRequest
from telethon.tl.functions.messages import GetFullChatRequest

import db as db_module
import notifier
from join_chats import (
    AuthError,
    ConfigError,
    EXIT_AUTH_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_NO_CHATS,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    build_proxy,
    configure_logging,
    load_environment,
    load_queries,
    normalize_target,
    read_api_credentials,
    read_session_name,
)

LOGGER = logging.getLogger("lead_sniper.auto_discover")

#: Signature of an awaitable sleep, injected in tests to skip real waiting.
Sleeper = Callable[[float], Awaitable[None]]

# --------------------------------------------------------------------------- #
# Environment configuration
# --------------------------------------------------------------------------- #

#: Minimum ``participants_count`` a candidate must report (default 50).
MIN_MEMBERS_ENV: str = "DISCOVERY_MIN_MEMBERS"
#: Hard cap of joins per UTC day (default 5).
DAILY_JOIN_LIMIT_ENV: str = "DISCOVERY_DAILY_JOIN_LIMIT"
#: How old the newest message may be, in hours (default 48).
MAX_AGE_HOURS_ENV: str = "DISCOVERY_MAX_AGE_HOURS"
#: Minimum number of messages inside the 7-day velocity window (default 5).
MIN_MESSAGES_ENV: str = "DISCOVERY_MIN_MESSAGES_7D"
#: How many recent messages are fetched for the activity audit (default 10).
HISTORY_LIMIT_ENV: str = "DISCOVERY_HISTORY_LIMIT"

DEFAULT_MIN_MEMBERS: int = 50
DEFAULT_DAILY_JOIN_LIMIT: int = 5
DEFAULT_MAX_AGE_HOURS: float = 48.0
DEFAULT_MIN_MESSAGES_7D: int = 5
DEFAULT_HISTORY_LIMIT: int = 10

#: Random jitter applied *between* search queries, seconds.
DEFAULT_SEARCH_DELAY_RANGE: tuple[float, float] = (10.0, 25.0)
#: Random jitter applied *between* actual join requests, seconds.
DEFAULT_JOIN_DELAY_RANGE: tuple[float, float] = (180.0, 360.0)

#: Velocity window for the activity audit.
VELOCITY_WINDOW: timedelta = timedelta(days=7)

#: Maximum number of directory hits requested per keyword.
SEARCH_RESULT_LIMIT: int = 20

#: Statuses that make a chat "done": it is never evaluated again.
SKIP_STATUSES: frozenset[str] = frozenset(
    {
        db_module.STATUS_JOINED,
        db_module.STATUS_ALREADY_JOINED,
        db_module.STATUS_BLACKLISTED,
        db_module.STATUS_TOO_FEW_MEMBERS,
    }
)


def _read_env(name: str) -> str:
    """Return a stripped environment value, or an empty string when unset."""
    return (os.getenv(name) or "").strip()


def read_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an integer environment variable with validation and a default.

    A blank value yields ``default``; a non-numeric value is logged and also
    yields ``default``; a value below ``minimum`` is clamped.
    """
    raw = _read_env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning(
            "%s=%r is not an integer; falling back to %d.", name, raw, default
        )
        return default
    if value < minimum:
        LOGGER.warning(
            "%s=%d is below the minimum %d; clamping to %d.",
            name,
            value,
            minimum,
            minimum,
        )
        return minimum
    return value


def read_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a float environment variable with validation and a default."""
    raw = _read_env(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        LOGGER.warning(
            "%s=%r is not a number; falling back to %s.", name, raw, default
        )
        return default
    if value < minimum:
        LOGGER.warning(
            "%s=%s is below the minimum %s; clamping to %s.",
            name,
            value,
            minimum,
            minimum,
        )
        return minimum
    return value


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Immutable discovery thresholds resolved from the environment.

    Attributes:
        min_members: Minimum ``participants_count`` for a candidate.
        daily_join_limit: Hard cap of joins per UTC day.
        max_age_hours: Maximum age of the newest message, in hours.
        min_messages_7d: Minimum messages inside the 7-day velocity window.
        history_limit: Number of recent messages fetched for the audit.
        search_delay_range: ``(min, max)`` jitter between search queries.
        join_delay_range: ``(min, max)`` jitter between join requests.
    """

    min_members: int = DEFAULT_MIN_MEMBERS
    daily_join_limit: int = DEFAULT_DAILY_JOIN_LIMIT
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS
    min_messages_7d: int = DEFAULT_MIN_MESSAGES_7D
    history_limit: int = DEFAULT_HISTORY_LIMIT
    search_delay_range: tuple[float, float] = DEFAULT_SEARCH_DELAY_RANGE
    join_delay_range: tuple[float, float] = DEFAULT_JOIN_DELAY_RANGE


def load_config() -> DiscoveryConfig:
    """Build a :class:`DiscoveryConfig` from the current environment."""
    return DiscoveryConfig(
        min_members=read_int_env(
            MIN_MEMBERS_ENV, DEFAULT_MIN_MEMBERS, minimum=1
        ),
        daily_join_limit=read_int_env(
            DAILY_JOIN_LIMIT_ENV, DEFAULT_DAILY_JOIN_LIMIT, minimum=0
        ),
        max_age_hours=read_float_env(
            MAX_AGE_HOURS_ENV, DEFAULT_MAX_AGE_HOURS, minimum=0.0
        ),
        min_messages_7d=read_int_env(
            MIN_MESSAGES_ENV, DEFAULT_MIN_MESSAGES_7D, minimum=0
        ),
        history_limit=read_int_env(
            HISTORY_LIMIT_ENV, DEFAULT_HISTORY_LIMIT, minimum=1
        ),
    )


# --------------------------------------------------------------------------- #
# Evaluation model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ChatEvaluation:
    """Outcome of the pre-join audit for a single candidate chat.

    Attributes:
        chat_id: Numeric Telegram id (``0`` when the entity carries none).
        username: Public username without ``@``.
        title: Human-readable chat title.
        members_count: ``participants_count`` reported by Telegram, if any.
        last_message_date: Timestamp of the newest observed message.
        status: One of :data:`db.KNOWN_STATUSES`.
        reason: Human-readable explanation of the verdict.
        messages_7d: Messages observed inside the 7-day velocity window.
        unique_senders: Distinct senders inside that window.
        entity: Resolved Telethon entity, kept for the join call. Never
            persisted and excluded from equality checks.
    """

    chat_id: int
    username: str
    title: str
    members_count: Optional[int]
    last_message_date: Optional[datetime]
    status: str
    reason: str
    messages_7d: int = 0
    unique_senders: int = 0
    entity: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        """``True`` when the chat cleared every pre-join gate."""
        return self.status == db_module.STATUS_ELIGIBLE


def _as_utc(value: object) -> Optional[datetime]:
    """Return ``value`` as a timezone-aware UTC datetime, or ``None``."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _chat_fields(entity: object) -> tuple[int, str, str, Optional[int]]:
    """Return ``(chat_id, username, title, participants_count)`` for ``entity``.

    Missing or malformed attributes degrade to ``0`` / ``""`` / ``None``
    instead of raising, so a partially populated search result cannot crash
    the discovery run.
    """
    raw_id = getattr(entity, "id", None)
    chat_id = (
        raw_id
        if isinstance(raw_id, int) and not isinstance(raw_id, bool)
        else 0
    )
    username = str(getattr(entity, "username", "") or "").strip()
    title = str(getattr(entity, "title", "") or "").strip()
    members = getattr(entity, "participants_count", None)
    if isinstance(members, bool) or not isinstance(members, int):
        members = None
    return chat_id, username, title, members


def _make_evaluation(
    entity: object,
    *,
    status: str,
    reason: str,
    last_message_date: Optional[datetime] = None,
    messages_7d: int = 0,
    unique_senders: int = 0,
) -> ChatEvaluation:
    """Build a :class:`ChatEvaluation` from a Telethon entity and a verdict."""
    chat_id, username, title, members = _chat_fields(entity)
    return ChatEvaluation(
        chat_id=chat_id,
        username=username,
        title=title,
        members_count=members,
        last_message_date=last_message_date,
        status=status,
        reason=reason,
        messages_7d=messages_7d,
        unique_senders=unique_senders,
        entity=entity,
    )


def is_joinable_entity(entity: object) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the structural entity gate.

    Accepts megagroups/supergroups, broadcast channels that expose a linked
    discussion group, and basic groups (which carry neither flag). Rejects
    deactivated chats, chats without a public username and broadcast-only
    channels where clients cannot post.
    """
    if bool(getattr(entity, "deactivated", False)):
        return False, "chat is deactivated"

    username = getattr(entity, "username", None)
    if not isinstance(username, str) or not username.strip():
        return False, "chat has no public username"

    if bool(getattr(entity, "megagroup", False)):
        return True, ""

    if bool(getattr(entity, "broadcast", False)):
        if bool(getattr(entity, "comments_enabled", False)):
            return True, ""
        return False, "broadcast channel without a discussion group"

    # A basic group (``Chat``) exposes neither ``megagroup`` nor ``broadcast``.
    return True, ""


def is_member(entity: object) -> bool:
    """Return ``True`` when the account is already a participant.

    Telethon sets ``left is False`` for chats the account has not left, i.e.
    it is a member; a missing ``left`` attribute means "unknown".
    """
    return getattr(entity, "left", None) is False


def evaluate_static(
    entity: object,
    *,
    config: Optional[DiscoveryConfig] = None,
    now: Optional[datetime] = None,
) -> Optional[ChatEvaluation]:
    """Run the gates that need no message history.

    Returns:
        A failing :class:`ChatEvaluation` when the entity gate, the membership
        check or the member-count gate rejects the chat, or ``None`` when the
        chat is eligible for the activity audit.
    """
    del now  # accepted for symmetry with ``evaluate_chat``; not needed here
    cfg = config or load_config()

    joinable, reason = is_joinable_entity(entity)
    if not joinable:
        return _make_evaluation(
            entity, status=db_module.STATUS_BLACKLISTED, reason=reason
        )

    if is_member(entity):
        return _make_evaluation(
            entity,
            status=db_module.STATUS_ALREADY_JOINED,
            reason="account is already a member",
        )

    _, _, _, members = _chat_fields(entity)
    if members is None:
        return _make_evaluation(
            entity,
            status=db_module.STATUS_TOO_FEW_MEMBERS,
            reason="participants count is unknown",
        )
    if members < cfg.min_members:
        return _make_evaluation(
            entity,
            status=db_module.STATUS_TOO_FEW_MEMBERS,
            reason=f"{members} members < {cfg.min_members} required",
        )
    return None


def _is_unknown_member_count(evaluation: ChatEvaluation) -> bool:
    """Return ``True`` when the member gate failed only for a missing count."""
    return (
        evaluation.status == db_module.STATUS_TOO_FEW_MEMBERS
        and evaluation.members_count is None
    )


def _usable_messages(
    messages: Optional[Sequence[object]],
) -> list[tuple[Optional[int], datetime]]:
    """Return ``(sender_id, date)`` pairs for real, dated messages.

    Service messages (joins, pins, …) carry an ``action`` attribute and are
    ignored: they inflate the velocity metric without representing any seller
    activity.
    """
    usable: list[tuple[Optional[int], datetime]] = []
    for message in messages or ():
        if getattr(message, "action", None) is not None:
            continue
        moment = _as_utc(getattr(message, "date", None))
        if moment is None:
            continue
        sender = getattr(message, "sender_id", None)
        sender_id = (
            sender
            if isinstance(sender, int) and not isinstance(sender, bool)
            else None
        )
        usable.append((sender_id, moment))
    return usable


def evaluate_chat(
    entity: object,
    messages: Optional[Sequence[object]],
    *,
    config: Optional[DiscoveryConfig] = None,
    now: Optional[datetime] = None,
) -> ChatEvaluation:
    """Evaluate a candidate against every pre-join gate (pure, synchronous).

    Args:
        entity: Telethon entity (or a stand-in) of the candidate chat.
        messages: The most recent messages of the chat, newest first or not.
        config: Thresholds; defaults to :func:`load_config`.
        now: Reference timestamp; defaults to the current UTC time.

    Returns:
        A :class:`ChatEvaluation` whose ``status`` is
        :data:`db.STATUS_ELIGIBLE` only when every gate passed.
    """
    cfg = config or load_config()
    reference = now or datetime.now(timezone.utc)

    static = evaluate_static(entity, config=cfg, now=reference)
    if static is not None:
        return static

    history = _usable_messages(messages)
    if not history:
        return _make_evaluation(
            entity,
            status=db_module.STATUS_INACTIVE,
            reason="no messages in the fetched history",
        )

    newest = max(moment for _, moment in history)
    age_hours = (reference - newest).total_seconds() / 3600.0
    if age_hours > cfg.max_age_hours:
        return _make_evaluation(
            entity,
            status=db_module.STATUS_INACTIVE,
            reason=(
                f"last message is {age_hours:.1f}h old "
                f"(> {cfg.max_age_hours:g}h limit)"
            ),
            last_message_date=newest,
        )

    window_start = reference - VELOCITY_WINDOW
    recent = [
        (sender, moment)
        for sender, moment in history
        if moment >= window_start
    ]
    messages_7d = len(recent)
    senders = {sender for sender, _ in recent if sender is not None}

    if messages_7d < cfg.min_messages_7d:
        return _make_evaluation(
            entity,
            status=db_module.STATUS_INACTIVE,
            reason=(
                f"only {messages_7d} message(s) in the last 7 days "
                f"(< {cfg.min_messages_7d} required)"
            ),
            last_message_date=newest,
            messages_7d=messages_7d,
            unique_senders=len(senders),
        )

    if messages_7d > 1 and len(senders) < 2:
        return _make_evaluation(
            entity,
            status=db_module.STATUS_INACTIVE,
            reason=(
                f"all {messages_7d} messages come from a single sender/bot"
            ),
            last_message_date=newest,
            messages_7d=messages_7d,
            unique_senders=len(senders),
        )

    return _make_evaluation(
        entity,
        status=db_module.STATUS_ELIGIBLE,
        reason=(
            f"active: {messages_7d} msg/7d from {len(senders)} sender(s), "
            f"last message {age_hours:.1f}h ago"
        ),
        last_message_date=newest,
        messages_7d=messages_7d,
        unique_senders=len(senders),
    )


async def validate_chat(
    client: TelegramClient,
    entity: object,
    *,
    config: Optional[DiscoveryConfig] = None,
    now: Optional[datetime] = None,
) -> ChatEvaluation:
    """Audit ``entity`` end to end, fetching its recent history.

    The static gates run first so a rejected chat costs no extra request. When
    the member gate cannot decide because the minimal entity carries no
    ``participants_count``, the full channel/chat metadata is fetched and the
    gates are re-run. When the history cannot be fetched (flood wait or network
    error) the chat is recorded as ``inactive`` — a false negative is always
    safe, a false positive is not.
    """
    cfg = config or load_config()
    reference = now or datetime.now(timezone.utc)

    static = evaluate_static(entity, config=cfg, now=reference)
    if static is not None:
        if not _is_unknown_member_count(static):
            return static

        try:
            members = await resolve_participants_count(client, entity)
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            LOGGER.warning(
                "Flood wait of %ss while resolving full info of %s; "
                "skipping chat.",
                seconds,
                _target_label(entity),
            )
            return _make_evaluation(
                entity,
                status=db_module.STATUS_INACTIVE,
                reason=f"flood wait of {seconds}s while resolving full chat",
            )

        if members is None:
            return static

        _apply_participants_count(entity, members)
        static = evaluate_static(entity, config=cfg, now=reference)
        if static is not None:
            return static

    try:
        messages = await client.get_messages(
            entity, limit=cfg.history_limit
        )
    except FloodWaitError as exc:
        seconds = int(getattr(exc, "seconds", 0) or 0)
        LOGGER.warning(
            "Flood wait of %ss while reading history of %s; skipping chat.",
            seconds,
            _target_label(entity),
        )
        return _make_evaluation(
            entity,
            status=db_module.STATUS_INACTIVE,
            reason=f"flood wait of {seconds}s while reading history",
        )
    except Exception as exc:  # noqa: BLE001 - one bad chat must not abort the scan
        LOGGER.warning(
            "Could not read history of %s: %s",
            _target_label(entity),
            exc,
        )
        return _make_evaluation(
            entity,
            status=db_module.STATUS_INACTIVE,
            reason=f"history unavailable: {exc}",
        )

    return evaluate_chat(entity, messages, config=cfg, now=reference)


def _target_label(entity: object) -> str:
    """Return a log-friendly ``@username`` (or numeric id) for ``entity``."""
    chat_id, username, title, _ = _chat_fields(entity)
    if username:
        return f"@{username}"
    return f"{title or 'chat'} (id={chat_id})"


def _valid_member_count(value: object) -> Optional[int]:
    """Return ``value`` as a non-negative participant count, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _full_chat_participants_count(full: object) -> Optional[int]:
    """Extract a participant count from a ``messages.ChatFull`` response.

    ``ChannelFull`` exposes ``participants_count`` directly, while a basic-group
    ``ChatFull`` only carries the participant list, so its length is used.
    """
    full_chat = getattr(full, "full_chat", full)
    members = _valid_member_count(
        getattr(full_chat, "participants_count", None)
    )
    if members is not None:
        return members
    participants = getattr(full_chat, "participants", None)
    listed = getattr(participants, "participants", None)
    if isinstance(listed, (list, tuple)):
        return len(listed)
    return None


def _apply_participants_count(entity: object, members: int) -> None:
    """Store a resolved participant count on ``entity`` when it is writable."""
    try:
        setattr(entity, "participants_count", members)
    except (AttributeError, TypeError):
        LOGGER.debug(
            "Could not store participants_count on %s.", _target_label(entity)
        )


async def resolve_participants_count(
    client: TelegramClient,
    entity: object,
) -> Optional[int]:
    """Return a trustworthy participant count for ``entity``.

    ``get_entity(username)`` frequently yields a minimal ``Channel`` whose
    ``participants_count`` is ``None``. In that case the full channel/chat
    metadata is requested so the member gate can make a real decision instead
    of rejecting an active chat as "unknown".

    The helper never raises when the full info is unavailable (private or
    restricted groups): it logs the failure and returns ``None`` so the caller
    records a safe false negative. ``FloodWaitError`` is propagated so the
    caller can stop the scan.
    """
    members = _valid_member_count(
        getattr(entity, "participants_count", None)
    )
    if members is not None:
        return members

    label = _target_label(entity)

    try:
        full = await client(GetFullChannelRequest(channel=entity))
    except FloodWaitError:
        raise
    except Exception as exc:  # noqa: BLE001 - full info is best-effort
        LOGGER.debug("Full channel lookup failed for %s: %s", label, exc)
    else:
        members = _full_chat_participants_count(full)
        if members is not None:
            LOGGER.debug(
                "Resolved %s members from full channel: %d", label, members
            )
            return members

    chat_id = getattr(entity, "id", None)
    if isinstance(chat_id, bool) or not isinstance(chat_id, int):
        return None

    try:
        full = await client(GetFullChatRequest(chat_id=chat_id))
    except FloodWaitError:
        raise
    except Exception as exc:  # noqa: BLE001 - full info is best-effort
        LOGGER.debug("Full chat lookup failed for %s: %s", label, exc)
        return None

    members = _full_chat_participants_count(full)
    if members is not None:
        LOGGER.debug("Resolved %s members from full chat: %d", label, members)
    return members


def describe_evaluation(evaluation: ChatEvaluation) -> str:
    """Render one audit line: title, members, last message, verdict, reason."""
    last_seen = (
        evaluation.last_message_date.isoformat()
        if evaluation.last_message_date is not None
        else "n/a"
    )
    members = (
        str(evaluation.members_count)
        if evaluation.members_count is not None
        else "?"
    )
    verdict = "PASS" if evaluation.passed else f"FAIL[{evaluation.status}]"
    return (
        f"@{evaluation.username or evaluation.chat_id} | "
        f"{evaluation.title or '(no title)'} | members={members} | "
        f"last_message={last_seen} | {evaluation.messages_7d} msg/7d, "
        f"{evaluation.unique_senders} sender(s) | {verdict}: "
        f"{evaluation.reason}"
    )


# --------------------------------------------------------------------------- #
# Anti-flood jitter
# --------------------------------------------------------------------------- #


def search_delay(
    config: Optional[DiscoveryConfig] = None,
    *,
    rng: Optional[random.Random] = None,
) -> float:
    """Return the random pause to apply before the next search query."""
    cfg = config or load_config()
    generator = rng or random
    return generator.uniform(*cfg.search_delay_range)


def join_delay(
    config: Optional[DiscoveryConfig] = None,
    *,
    rng: Optional[random.Random] = None,
) -> float:
    """Return the random pause to apply before the next join request."""
    cfg = config or load_config()
    generator = rng or random
    return generator.uniform(*cfg.join_delay_range)


# --------------------------------------------------------------------------- #
# Search & collection
# --------------------------------------------------------------------------- #


async def search_usernames(
    client: TelegramClient,
    keywords: Sequence[str],
    *,
    config: Optional[DiscoveryConfig] = None,
    sleep: Optional[Sleeper] = None,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """Search Telegram's public directory for ``keywords``.

    A 10-25 s jitter is applied *between* queries (never before the first).
    Failing or flood-limited queries are logged and skipped so one bad keyword
    cannot abort the run. Usernames are normalised and deduplicated while
    preserving discovery order.
    """
    cfg = config or load_config()
    sleeper = sleep or asyncio.sleep
    generator = rng or random.Random()
    found: list[str] = []
    queried = 0

    for raw in keywords:
        keyword = (raw or "").strip()
        if not keyword:
            continue

        if queried and cfg.search_delay_range[1] > 0:
            delay = search_delay(cfg, rng=generator)
            LOGGER.info(
                "Sleeping %.1fs (anti-flood) before searching %r...",
                delay,
                keyword,
            )
            await sleeper(delay)
        queried += 1

        try:
            result = await client(
                SearchRequest(q=keyword, limit=SEARCH_RESULT_LIMIT)
            )
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            LOGGER.warning(
                "Flood wait of %ss while searching %r; skipping the query.",
                seconds,
                keyword,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - one bad query must not abort
            LOGGER.warning("Search for %r failed: %s", keyword, exc)
            continue

        for chat in getattr(result, "chats", None) or []:
            username = normalize_target(
                str(getattr(chat, "username", "") or "")
            )
            if not username or username in found:
                continue
            found.append(username)
            LOGGER.info(
                "Discovered @%s (%s) via %r.",
                username,
                getattr(chat, "title", None) or "(no title)",
                keyword,
            )

    LOGGER.info(
        "Discovered %d candidate(s) across %d search quer(y/ies).",
        len(found),
        queried,
    )
    return found


async def collect_evaluations(
    client: TelegramClient,
    usernames: Sequence[str],
    *,
    config: Optional[DiscoveryConfig] = None,
    now: Optional[datetime] = None,
    db_path: str = db_module.DEFAULT_DB_PATH,
    console: bool = False,
) -> list[ChatEvaluation]:
    """Resolve, audit and persist every candidate in ``usernames``.

    Chats already recorded with a "done" status (joined, already joined,
    blacklisted, too few members) are skipped without a history request, so a
    repeated run costs almost nothing. Each new verdict is written to the
    ``discovered_chats`` audit table.

    Args:
        client: Connected Telethon client.
        usernames: Bare usernames to process, in order.
        config: Thresholds; defaults to :func:`load_config`.
        now: Reference timestamp shared by every evaluation.
        db_path: Audit SQLite file.
        console: Also print each audit line (used by ``--dry-run``).
    """
    cfg = config or load_config()
    reference = now or datetime.now(timezone.utc)
    evaluations: list[ChatEvaluation] = []

    for username in usernames:
        try:
            entity = await client.get_entity(username)
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            LOGGER.warning(
                "Flood wait of %ss while resolving @%s; stopping the scan.",
                seconds,
                username,
            )
            break
        except Exception as exc:  # noqa: BLE001 - one bad username is skippable
            LOGGER.warning("Could not resolve @%s: %s", username, exc)
            continue

        chat_id, _, _, _ = _chat_fields(entity)
        if chat_id and await db_module.is_chat_evaluated(chat_id, db_path):
            status = await db_module.get_chat_status(chat_id, db_path)
            if status in SKIP_STATUSES:
                LOGGER.info(
                    "Skipping @%s: already evaluated (status=%s).",
                    username,
                    status,
                )
                continue

        evaluation = await validate_chat(
            client, entity, config=cfg, now=reference
        )
        await db_module.record_chat_evaluation(
            chat_id=evaluation.chat_id,
            username=evaluation.username,
            title=evaluation.title,
            members_count=evaluation.members_count,
            last_message_date=evaluation.last_message_date,
            status=evaluation.status,
            db_path=db_path,
        )
        evaluations.append(evaluation)

        line = describe_evaluation(evaluation)
        LOGGER.info("Audit: %s", line)
        if console:
            print(f"[?] {line}")

    return evaluations


# --------------------------------------------------------------------------- #
# Joining
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class JoinStats:
    """Aggregated outcome of a join pass."""

    evaluated: int = 0
    joined: list[ChatEvaluation] = field(default_factory=list)
    already_joined: list[ChatEvaluation] = field(default_factory=list)
    rejected: list[ChatEvaluation] = field(default_factory=list)
    failed: list[ChatEvaluation] = field(default_factory=list)
    skipped_due_to_limit: list[ChatEvaluation] = field(default_factory=list)
    joined_today: int = 0
    limit_reached: bool = False
    account_limit: bool = False
    flood_wait_seconds: int = 0
    dry_run: bool = False


async def _persist(
    evaluation: ChatEvaluation,
    *,
    status: str,
    joined: bool,
    db_path: str,
) -> None:
    """Write the (possibly new) verdict of ``evaluation`` to the audit table."""
    await db_module.record_chat_evaluation(
        chat_id=evaluation.chat_id,
        username=evaluation.username,
        title=evaluation.title,
        members_count=evaluation.members_count,
        last_message_date=evaluation.last_message_date,
        status=status,
        joined=joined,
        db_path=db_path,
    )
    evaluation.status = status


async def join_eligible_chats(
    client: TelegramClient,
    evaluations: Sequence[ChatEvaluation],
    *,
    config: Optional[DiscoveryConfig] = None,
    dry_run: bool = False,
    join_limit: Optional[int] = None,
    db_path: str = db_module.DEFAULT_DB_PATH,
    sleep: Optional[Sleeper] = None,
    rng: Optional[random.Random] = None,
) -> JoinStats:
    """Join the eligible chats, honouring both limits and the anti-flood rules.

    The effective per-run cap is ``min(join_limit, daily_join_limit)`` and is
    further reduced by the number of joins already recorded today. A 180-360 s
    jitter is inserted before every ``JoinChannelRequest`` after the first one.

    Args:
        client: Connected Telethon client.
        evaluations: Audited candidates, in discovery order.
        config: Thresholds; defaults to :func:`load_config`.
        dry_run: Audit only — ``JoinChannelRequest`` is never sent.
        join_limit: Per-run cap from ``--limit``; ``None`` means "daily limit".
        db_path: Audit SQLite file.
        sleep: Awaitable used for jitter (injectable for tests).
        rng: Random generator used for jitter (injectable for tests).
    """
    cfg = config or load_config()
    sleeper = sleep or asyncio.sleep
    generator = rng or random.Random()
    stats = JoinStats(evaluated=len(evaluations), dry_run=dry_run)

    if dry_run:
        for evaluation in evaluations:
            if evaluation.passed:
                LOGGER.info("DRY-RUN pass: %s", describe_evaluation(evaluation))
            else:
                LOGGER.info(
                    "DRY-RUN reject: %s", describe_evaluation(evaluation)
                )
                stats.rejected.append(evaluation)
        return stats

    joined_today = await db_module.get_daily_joined_count(db_path)
    stats.joined_today = joined_today
    if joined_today >= cfg.daily_join_limit:
        LOGGER.warning(
            "Daily join limit reached (%d/%d) — no joins today.",
            joined_today,
            cfg.daily_join_limit,
        )
        stats.limit_reached = True
        stats.skipped_due_to_limit.extend(
            evaluation for evaluation in evaluations if evaluation.passed
        )
        return stats

    run_cap = (
        cfg.daily_join_limit
        if join_limit is None
        else min(max(join_limit, 0), cfg.daily_join_limit)
    )
    remaining = min(run_cap, cfg.daily_join_limit - joined_today)
    attempts = 0

    for evaluation in evaluations:
        if not evaluation.passed:
            stats.rejected.append(evaluation)
            continue

        if remaining <= 0:
            stats.limit_reached = True
            stats.skipped_due_to_limit.append(evaluation)
            continue

        if attempts and cfg.join_delay_range[1] > 0:
            delay = join_delay(cfg, rng=generator)
            LOGGER.info(
                "Sleeping %.0fs (anti-flood jitter) before joining @%s...",
                delay,
                evaluation.username,
            )
            await sleeper(delay)
        attempts += 1

        try:
            if evaluation.entity is not None:
                await client(JoinChannelRequest(evaluation.entity))
            else:
                resolved = await client.get_entity(evaluation.username)
                await client(JoinChannelRequest(resolved))
        except UserAlreadyParticipantError:
            LOGGER.info(
                "Already a member of @%s; recording it.", evaluation.username
            )
            await _persist(
                evaluation,
                status=db_module.STATUS_ALREADY_JOINED,
                joined=False,
                db_path=db_path,
            )
            stats.already_joined.append(evaluation)
            continue
        except ChannelsTooMuchError as exc:
            LOGGER.error(
                "Account channel limit reached (%s); stopping joins.", exc
            )
            stats.account_limit = True
            break
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            LOGGER.warning(
                "Flood wait of %ss while joining @%s; stopping the session.",
                seconds,
                evaluation.username,
            )
            stats.flood_wait_seconds = seconds
            await _persist(
                evaluation,
                status=evaluation.status,
                joined=False,
                db_path=db_path,
            )
            break
        except Exception as exc:  # noqa: BLE001 - one bad chat is skippable
            LOGGER.warning(
                "Join failed for @%s: %s", evaluation.username, exc
            )
            stats.failed.append(evaluation)
            continue

        await _persist(
            evaluation,
            status=db_module.STATUS_JOINED,
            joined=True,
            db_path=db_path,
        )
        stats.joined.append(evaluation)
        remaining -= 1
        LOGGER.info(
            "Joined @%s (%s).",
            evaluation.username,
            evaluation.title or "no title",
        )

    stats.joined_today += len(stats.joined)
    return stats


# --------------------------------------------------------------------------- #
# Session orchestration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SessionSummary:
    """Result of one discovery session (used for the Telegram summary)."""

    keywords: int = 0
    discovered: int = 0
    evaluations: list[ChatEvaluation] = field(default_factory=list)
    stats: JoinStats = field(default_factory=JoinStats)
    dry_run: bool = False
    limit_reached: bool = False


async def daily_join_quota(
    config: Optional[DiscoveryConfig] = None,
    db_path: str = db_module.DEFAULT_DB_PATH,
) -> int:
    """Return how many joins are still allowed today (``>= 0``)."""
    cfg = config or load_config()
    joined_today = await db_module.get_daily_joined_count(db_path)
    return max(cfg.daily_join_limit - joined_today, 0)


async def run_session(
    client: TelegramClient,
    keywords: Sequence[str],
    *,
    config: Optional[DiscoveryConfig] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    db_path: str = db_module.DEFAULT_DB_PATH,
    now: Optional[datetime] = None,
    sleep: Optional[Sleeper] = None,
    rng: Optional[random.Random] = None,
    console: bool = False,
) -> SessionSummary:
    """Search, audit and join in one pass, returning the session summary.

    In a live run the daily join limit is checked *before* any network call; a
    reached limit yields an empty summary flagged with ``limit_reached``.
    """
    cfg = config or load_config()
    await db_module.init_db(db_path)
    summary = SessionSummary(keywords=len(keywords), dry_run=dry_run)

    if not dry_run and await daily_join_quota(cfg, db_path) <= 0:
        joined_today = await db_module.get_daily_joined_count(db_path)
        LOGGER.warning(
            "Daily join limit reached (%d/%d) — exiting gracefully.",
            joined_today,
            cfg.daily_join_limit,
        )
        summary.limit_reached = True
        summary.stats.limit_reached = True
        summary.stats.joined_today = joined_today
        return summary

    usernames = await search_usernames(
        client, keywords, config=cfg, sleep=sleep, rng=rng
    )
    summary.discovered = len(usernames)
    summary.evaluations = await collect_evaluations(
        client,
        usernames,
        config=cfg,
        now=now,
        db_path=db_path,
        console=console or dry_run,
    )
    summary.stats = await join_eligible_chats(
        client,
        summary.evaluations,
        config=cfg,
        dry_run=dry_run,
        join_limit=limit,
        db_path=db_path,
        sleep=sleep,
        rng=rng,
    )
    summary.stats.joined_today = await db_module.get_daily_joined_count(db_path)
    summary.limit_reached = summary.stats.limit_reached
    return summary


def format_summary(summary: SessionSummary) -> str:
    """Render the session summary as a Telegram HTML message body."""
    stats = summary.stats
    status_counts: dict[str, int] = {}
    for evaluation in summary.evaluations:
        status_counts[evaluation.status] = (
            status_counts.get(evaluation.status, 0) + 1
        )

    lines: list[str] = ["🔎 <b>Smart Discovery — итоги сессии</b>"]
    if summary.dry_run:
        lines.append("<i>Режим: dry-run (без вступления)</i>")
    lines.append(
        f"Ключевых слов: {summary.keywords} | Найдено: "
        f"{summary.discovered} | Проверено: {len(summary.evaluations)}"
    )
    lines.append(
        f"Новых джойнов: {len(stats.joined)} "
        f"(за сутки: {stats.joined_today})"
    )
    if stats.already_joined:
        lines.append(f"Уже участник: {len(stats.already_joined)}")
    if stats.failed:
        lines.append(f"Ошибок джойна: {len(stats.failed)}")
    if status_counts:
        breakdown = ", ".join(
            f"{status}={count}"
            for status, count in sorted(status_counts.items())
        )
        lines.append(f"Статусы: {html.escape(breakdown)}")
    if stats.limit_reached:
        lines.append("⚠️ Достигнут дневной лимит джойнов.")
    if stats.account_limit:
        lines.append("⛔ Лимит каналов аккаунта исчерпан.")
    if stats.flood_wait_seconds:
        lines.append(f"⏳ FloodWait: {stats.flood_wait_seconds}s.")

    return "\n".join(lines)[: notifier.MAX_MESSAGE_LENGTH]


async def send_summary(
    summary: SessionSummary,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
) -> bool:
    """Deliver the session summary to ``TELEGRAM_ALERT_CHAT_ID``.

    Returns ``False`` (never raises) when the bot is not configured or the
    request fails: a notification problem must not fail the discovery run.
    """
    config = notifier.read_telegram_config()
    if not config.is_configured:
        LOGGER.info(
            "Summary notification skipped: TELEGRAM_BOT_TOKEN / "
            "TELEGRAM_ALERT_CHAT_ID are not configured."
        )
        return False

    url = f"{notifier.API_BASE}/bot{config.bot_token}/sendMessage"
    payload = {
        "chat_id": config.chat_id,
        "text": format_summary(summary),
        "parse_mode": notifier.PARSE_MODE,
        "disable_web_page_preview": True,
    }

    owns_client = http_client is None
    http: Optional[httpx.AsyncClient] = None
    try:
        if http_client is not None:
            http = http_client
        else:
            http = httpx.AsyncClient(
                proxy=notifier.resolve_proxy_url(), timeout=config.timeout
            )
        response = await http.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001 - notification is best-effort
        LOGGER.warning("Summary notification failed: %s", exc)
        return False
    finally:
        if owns_client and http is not None:
            await http.aclose()

    if response.status_code != 200:
        LOGGER.warning(
            "Summary notification rejected: HTTP %d %s",
            response.status_code,
            response.text[:200],
        )
        return False

    LOGGER.info("Summary notification delivered to %s.", config.chat_id)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def resolve_keywords(
    cli_keywords: Optional[Sequence[str]] = None,
) -> list[str]:
    """Return the effective keyword list for a run.

    Explicit ``--keyword`` values win; otherwise ``keywords.txt`` is used,
    falling back to the built-in query list inside :func:`join_chats.load_queries`.
    """
    if cli_keywords:
        cleaned: list[str] = []
        for raw in cli_keywords:
            keyword = (raw or "").strip()
            if keyword and keyword not in cleaned:
                cleaned.append(keyword)
        if cleaned:
            return cleaned
    return load_queries()


def _print_summary(summary: SessionSummary) -> None:
    """Print a compact, human-readable session recap."""
    stats = summary.stats
    print()
    print(
        f"[*] Keywords: {summary.keywords} | discovered: "
        f"{summary.discovered} | evaluated: {len(summary.evaluations)}"
    )
    print(
        f"[*] Joined: {len(stats.joined)} | already member: "
        f"{len(stats.already_joined)} | rejected: {len(stats.rejected)} | "
        f"failed: {len(stats.failed)}"
    )
    if stats.joined:
        print(
            "[+] New: " + ", ".join(e.username for e in stats.joined if e.username)
        )
    if stats.limit_reached:
        print("[!] Daily join limit reached; stopping.")
    if stats.account_limit:
        print("[-] Account channel limit reached; stopping.")
    if stats.flood_wait_seconds:
        print(f"[!] FloodWait {stats.flood_wait_seconds}s; session stopped.")


async def run(args: argparse.Namespace) -> int:
    """Connect, run one discovery session and optionally notify."""
    load_environment()
    config = load_config()
    keywords = resolve_keywords(getattr(args, "keyword", None))
    if not keywords:
        LOGGER.error(
            "No search keywords configured. Add them to keywords.txt or pass "
            "--keyword TEXT."
        )
        return EXIT_CONFIG_ERROR

    db_path = str(getattr(args, "db", db_module.DEFAULT_DB_PATH) or "")
    db_path = db_path or db_module.DEFAULT_DB_PATH
    dry_run = bool(getattr(args, "dry_run", False))

    if not dry_run:
        await db_module.init_db(db_path)
        if await daily_join_quota(config, db_path) <= 0:
            LOGGER.warning(
                "Daily join limit of %d join(s) is already reached for today — "
                "exiting gracefully.",
                config.daily_join_limit,
            )
            print(
                f"[!] Daily join limit ({config.daily_join_limit}) already "
                "reached; nothing to do."
            )
            return EXIT_OK

    api_id, api_hash = read_api_credentials()
    session_name = read_session_name()
    proxy = build_proxy()

    client = TelegramClient(session_name, api_id, api_hash, proxy=proxy)
    try:
        LOGGER.info(
            "Connecting to Telegram (session=%s.session)...", session_name
        )
        await client.connect()

        if not await client.is_user_authorized():
            raise AuthError(
                "Telegram session is not authorized. Run `python auth_test.py` "
                "first to link your account, then re-run this script."
            )

        summary = await run_session(
            client,
            keywords,
            config=config,
            dry_run=dry_run,
            limit=getattr(args, "limit", None),
            db_path=db_path,
        )
    finally:
        if client.is_connected():
            await client.disconnect()
            LOGGER.info("Disconnected from Telegram.")

    _print_summary(summary)

    if dry_run:
        print()
        print("[*] Dry-run complete; no JoinChannelRequest was sent.")
        return EXIT_OK

    if summary.stats.joined:
        await send_summary(summary)
        return EXIT_OK

    print()
    print("[-] No chats were joined in this run.")
    return EXIT_NO_CHATS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI flags for a discovery session."""
    parser = argparse.ArgumentParser(
        description=(
            "Discover public seller chats by keyword, audit their activity "
            "before joining and enforce strict anti-flood limits."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Search and audit every candidate (printing title, members, last "
            "message date and the pass/fail reason) without joining."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum successful joins for this run (default: the daily "
            "limit from DISCOVERY_DAILY_JOIN_LIMIT; never exceeds it)."
        ),
    )
    parser.add_argument(
        "--keyword",
        action="append",
        metavar="TEXT",
        default=None,
        help=(
            "Search keyword; repeatable. Defaults to keywords.txt, then the "
            "built-in query list."
        ),
    )
    parser.add_argument(
        "--db",
        default=db_module.DEFAULT_DB_PATH,
        metavar="PATH",
        help=(
            "Audit SQLite file "
            f"(default: {db_module.DEFAULT_DB_PATH})."
        ),
    )

    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be greater than or equal to 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Synchronous CLI wrapper translating failures into exit codes."""
    configure_logging()
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return EXIT_CONFIG_ERROR
    except AuthError as exc:
        LOGGER.error("%s", exc)
        return EXIT_AUTH_ERROR
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return EXIT_RUNTIME_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI
        LOGGER.error("Unexpected failure: %s", exc)
        LOGGER.debug("Traceback:", exc_info=True)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
