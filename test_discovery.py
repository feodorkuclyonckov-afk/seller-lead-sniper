"""Verification suite for :mod:`auto_discover` and the discovery audit in :mod:`db`.

Run with::

    python -m pytest test_discovery.py -v

Every network-facing dependency is mocked: the "Telethon client" is a small
stand-in exposing ``__call__`` / ``get_entity`` / ``get_messages``, entities and
messages are :class:`types.SimpleNamespace` objects, and the SQLite audit table
lives in ``tmp_path``. No credentials, proxy or network access are required.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
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

import auto_discover as ad
import db as db_module
import join_chats

#: Fixed reference timestamp so freshness/velocity gates are deterministic.
REFERENCE: datetime = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

#: Daily limit used by the join tests (small to keep scenarios readable).
TEST_LIMIT: int = 2

#: Config with production defaults but a shorter join jitter for clarity.
TEST_CONFIG: ad.DiscoveryConfig = ad.DiscoveryConfig(daily_join_limit=TEST_LIMIT)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


async def _noop_sleep(delay: float) -> None:
    """Async no-op used as the injectable sleeper in join/search loops."""
    del delay


def make_entity(
    chat_id: int = 1001,
    username: Optional[str] = "seller_chat",
    title: str = "Seller Chat",
    members: Optional[int] = 1500,
    *,
    megagroup: bool = True,
    broadcast: bool = False,
    deactivated: bool = False,
    left: Optional[bool] = True,
    comments_enabled: bool = False,
    with_flags: bool = True,
) -> SimpleNamespace:
    """Build a Telethon-like entity for the pre-join gates."""
    entity = SimpleNamespace(
        id=chat_id,
        username=username,
        title=title,
        participants_count=members,
        deactivated=deactivated,
    )
    entity.left = left
    if with_flags:
        entity.megagroup = megagroup
        entity.broadcast = broadcast
        entity.comments_enabled = comments_enabled
    return entity


def make_message(
    days_ago: float,
    sender_id: int = 1,
    *,
    is_service: bool = False,
) -> SimpleNamespace:
    """Build a Telethon-like message dated ``days_ago`` before the reference."""
    return SimpleNamespace(
        date=REFERENCE - timedelta(days=days_ago),
        sender_id=sender_id,
        action=SimpleNamespace() if is_service else None,
    )


def healthy_history() -> list[SimpleNamespace]:
    """Eight recent messages (last 3 days) from three distinct senders."""
    return [
        make_message(0.1 + index * 0.35, sender_id=1 + index % 3)
        for index in range(8)
    ]


def full_channel_response(members: int) -> SimpleNamespace:
    """Build a ``messages.ChatFull`` stand-in for a supergroup/channel."""
    return SimpleNamespace(
        full_chat=SimpleNamespace(participants_count=members)
    )


def full_chat_response(participant_ids: list[int]) -> SimpleNamespace:
    """Build a ``messages.ChatFull`` stand-in for a basic group."""
    return SimpleNamespace(
        full_chat=SimpleNamespace(
            participants=SimpleNamespace(
                participants=[
                    SimpleNamespace(user_id=user_id)
                    for user_id in participant_ids
                ]
            )
        )
    )


def eligible_evaluation(
    chat_id: int,
    username: str,
    *,
    members: int = 1500,
) -> ad.ChatEvaluation:
    """Return a passing evaluation for a healthy chat."""
    return ad.evaluate_chat(
        make_entity(chat_id=chat_id, username=username, members=members),
        healthy_history(),
        config=TEST_CONFIG,
        now=REFERENCE,
    )


class FakeClient:
    """Minimal stand-in for ``TelegramClient`` covering the calls we issue."""

    def __init__(
        self,
        *,
        chats: Optional[dict[str, list[Any]]] = None,
        entities: Optional[dict[str, Any]] = None,
        histories: Optional[dict[int, list[Any]]] = None,
        join_effect: Optional[BaseException] = None,
        full_channels: Optional[dict[int, Any]] = None,
        full_chats: Optional[dict[int, Any]] = None,
        full_channel_effect: Optional[BaseException] = None,
        full_chat_effect: Optional[BaseException] = None,
    ) -> None:
        self.chat_results = chats or {}
        self.entities = entities or {}
        self.histories = histories or {}
        self.join_effect = join_effect
        self.full_channels = full_channels or {}
        self.full_chats = full_chats or {}
        self.full_channel_effect = full_channel_effect
        self.full_chat_effect = full_chat_effect
        self.search_queries: list[str] = []
        self.join_requests: list[Any] = []
        self.full_channel_requests: list[Any] = []
        self.full_chat_requests: list[int] = []

    async def __call__(self, request: Any) -> Any:
        if isinstance(request, SearchRequest):
            self.search_queries.append(request.q)
            return SimpleNamespace(
                chats=list(self.chat_results.get(request.q, []))
            )
        if isinstance(request, GetFullChannelRequest):
            self.full_channel_requests.append(request.channel)
            if self.full_channel_effect is not None:
                raise self.full_channel_effect
            key = getattr(request.channel, "id", 0)
            if key not in self.full_channels:
                raise ValueError(f"no full channel for {key}")
            return self.full_channels[key]
        if isinstance(request, GetFullChatRequest):
            self.full_chat_requests.append(request.chat_id)
            if self.full_chat_effect is not None:
                raise self.full_chat_effect
            if request.chat_id not in self.full_chats:
                raise ValueError(f"no full chat for {request.chat_id}")
            return self.full_chats[request.chat_id]
        if isinstance(request, JoinChannelRequest):
            self.join_requests.append(request.channel)
            if self.join_effect is not None:
                raise self.join_effect
            return True
        raise AssertionError(f"Unexpected request: {request!r}")

    async def get_entity(self, username: str) -> Any:
        if username not in self.entities:
            raise ValueError(f"unknown entity: {username}")
        return self.entities[username]

    async def get_messages(self, entity: Any, limit: int = 10) -> list[Any]:
        del limit
        return list(self.histories.get(getattr(entity, "id", 0), []))


@pytest.fixture()
def audit_db(tmp_path: Path) -> str:
    """Return a fresh audit database path inside the test's temp directory."""
    return str(tmp_path / "discovery_audit.db")


# --------------------------------------------------------------------------- #
# Activity validation heuristics
# --------------------------------------------------------------------------- #


def test_abandoned_chat_fails_validation() -> None:
    """A chat whose newest message is older than 48h is rejected as inactive."""
    entity = make_entity(chat_id=2001, username="dead_chat")
    history = [
        make_message(3 + index, sender_id=10 + index) for index in range(6)
    ]

    result = ad.evaluate_chat(
        entity, history, config=ad.DiscoveryConfig(), now=REFERENCE
    )

    assert result.passed is False
    assert result.status == db_module.STATUS_INACTIVE
    assert "48" in result.reason
    assert result.last_message_date == REFERENCE - timedelta(days=3)


def test_low_member_group_fails_validation() -> None:
    """A group below DISCOVERY_MIN_MEMBERS is rejected before any join."""
    entity = make_entity(chat_id=2002, username="tiny_chat", members=12)

    result = ad.evaluate_chat(
        entity, healthy_history(), config=ad.DiscoveryConfig(), now=REFERENCE
    )

    assert result.passed is False
    assert result.status == db_module.STATUS_TOO_FEW_MEMBERS
    assert result.members_count == 12
    assert "12" in result.reason


def test_healthy_active_supergroup_passes_validation() -> None:
    """An active megagroup with enough members clears every gate."""
    entity = make_entity(chat_id=2003, username="live_chat", members=1500)

    result = ad.evaluate_chat(
        entity, healthy_history(), config=ad.DiscoveryConfig(), now=REFERENCE
    )

    assert result.passed is True
    assert result.status == db_module.STATUS_ELIGIBLE
    assert result.members_count == 1500
    assert result.messages_7d == 8
    assert result.unique_senders == 3
    assert result.last_message_date is not None
    assert result.entity is entity


def test_single_sender_chat_fails_validation() -> None:
    """Activity pumped by one sender/bot is not real activity."""
    history = [
        make_message(0.2 + index * 0.3, sender_id=42) for index in range(6)
    ]

    result = ad.evaluate_chat(
        make_entity(chat_id=2004),
        history,
        config=ad.DiscoveryConfig(),
        now=REFERENCE,
    )

    assert result.passed is False
    assert result.status == db_module.STATUS_INACTIVE
    assert "single sender" in result.reason
    assert result.unique_senders == 1


def test_low_velocity_chat_fails_validation() -> None:
    """Fewer than five messages inside 7 days is too quiet to join."""
    history = [
        make_message(0.5, sender_id=1),
        make_message(1.0, sender_id=2),
    ]

    result = ad.evaluate_chat(
        make_entity(chat_id=2005),
        history,
        config=ad.DiscoveryConfig(),
        now=REFERENCE,
    )

    assert result.status == db_module.STATUS_INACTIVE
    assert "last 7 days" in result.reason
    assert result.messages_7d == 2


def test_service_messages_do_not_count_as_activity() -> None:
    """Service messages (pins/joins) are ignored by the velocity gate."""
    history = [
        make_message(0.1, sender_id=1),
        make_message(0.2, sender_id=2),
        *[make_message(0.3 + i * 0.1, sender_id=3, is_service=True) for i in range(8)],
    ]

    result = ad.evaluate_chat(
        make_entity(chat_id=2006),
        history,
        config=ad.DiscoveryConfig(),
        now=REFERENCE,
    )

    assert result.status == db_module.STATUS_INACTIVE
    assert result.messages_7d == 2


def test_empty_history_fails_validation() -> None:
    """A chat that returns no messages at all is treated as inactive."""
    result = ad.evaluate_chat(
        make_entity(chat_id=2007),
        [],
        config=ad.DiscoveryConfig(),
        now=REFERENCE,
    )

    assert result.status == db_module.STATUS_INACTIVE
    assert "no messages" in result.reason


@pytest.mark.parametrize(
    ("entity", "expected_status"),
    [
        # Broadcast-only channel: clients cannot post there.
        (
            make_entity(
                chat_id=3001,
                username="news_channel",
                megagroup=False,
                broadcast=True,
            ),
            db_module.STATUS_BLACKLISTED,
        ),
        # Broadcast channel with a linked discussion group is acceptable.
        (
            make_entity(
                chat_id=3002,
                username="comments_channel",
                megagroup=False,
                broadcast=True,
                comments_enabled=True,
            ),
            db_module.STATUS_ELIGIBLE,
        ),
        # Deactivated chats can never be joined.
        (
            make_entity(chat_id=3003, deactivated=True),
            db_module.STATUS_BLACKLISTED,
        ),
        # No public username means the chat cannot be joined by link.
        (
            make_entity(chat_id=3004, username=None),
            db_module.STATUS_BLACKLISTED,
        ),
        # Already a member: recorded, never re-tried.
        (
            make_entity(chat_id=3005, left=False),
            db_module.STATUS_ALREADY_JOINED,
        ),
        # A basic group carries neither megagroup nor broadcast flags.
        (
            make_entity(chat_id=3006, with_flags=False),
            db_module.STATUS_ELIGIBLE,
        ),
    ],
)
def test_entity_gate(
    entity: SimpleNamespace, expected_status: str
) -> None:
    """Structural gates map to the documented audit statuses."""
    result = ad.evaluate_chat(
        entity, healthy_history(), config=ad.DiscoveryConfig(), now=REFERENCE
    )

    assert result.status == expected_status
    assert result.passed is (expected_status == db_module.STATUS_ELIGIBLE)


def test_unknown_participant_count_is_rejected() -> None:
    """A missing ``participants_count`` cannot be trusted as "big enough"."""
    result = ad.evaluate_chat(
        make_entity(chat_id=3007, members=None),
        healthy_history(),
        config=ad.DiscoveryConfig(),
        now=REFERENCE,
    )

    assert result.status == db_module.STATUS_TOO_FEW_MEMBERS
    assert "unknown" in result.reason


def test_validate_chat_marks_history_failure_inactive() -> None:
    """A history fetch failure is a safe false negative, never a pass."""
    entity = make_entity(chat_id=3100, username="broken_chat")

    class _BrokenClient(FakeClient):
        async def get_messages(self, entity: Any, limit: int = 10) -> list[Any]:
            del limit
            raise RuntimeError("history unavailable")

    result = asyncio.run(
        ad.validate_chat(_BrokenClient(), entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.status == db_module.STATUS_INACTIVE
    assert "history unavailable" in result.reason


def test_validate_chat_fetches_history_for_healthy_chat() -> None:
    """``validate_chat`` combines the static gates with the fetched history."""
    entity = make_entity(chat_id=3101, username="good_chat")
    client = FakeClient(histories={3101: healthy_history()})

    result = asyncio.run(
        ad.validate_chat(client, entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.passed is True
    assert result.entity is entity


# --------------------------------------------------------------------------- #
# Full channel resolution (member-gate false negatives)
# --------------------------------------------------------------------------- #


def test_validate_chat_resolves_full_channel_when_count_unknown() -> None:
    """A minimal ``Channel`` without ``participants_count`` is fully resolved."""
    entity = make_entity(chat_id=3200, username="unknown_count", members=None)
    client = FakeClient(
        histories={3200: healthy_history()},
        full_channels={3200: full_channel_response(1500)},
    )

    result = asyncio.run(
        ad.validate_chat(client, entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.passed is True
    assert result.status == db_module.STATUS_ELIGIBLE
    assert result.members_count == 1500
    assert client.full_channel_requests == [entity]
    assert client.full_chat_requests == []


def test_validate_chat_resolves_full_chat_for_basic_group() -> None:
    """A basic group falls back to ``GetFullChatRequest`` and counts members."""
    entity = make_entity(
        chat_id=3201,
        username="basic_group",
        members=None,
        with_flags=False,
    )
    client = FakeClient(
        histories={3201: healthy_history()},
        full_chats={3201: full_chat_response(list(range(60)))},
    )

    result = asyncio.run(
        ad.validate_chat(client, entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.passed is True
    assert result.members_count == 60
    assert client.full_channel_requests == [entity]
    assert client.full_chat_requests == [3201]


def test_validate_chat_marks_unknown_when_full_resolution_fails() -> None:
    """Private/restricted chats degrade to a safe false negative, never raise."""
    entity = make_entity(chat_id=3202, username="private_group", members=None)
    client = FakeClient(
        histories={3202: healthy_history()},
        full_channel_effect=RuntimeError("channel unavailable"),
        full_chat_effect=RuntimeError("chat unavailable"),
    )

    result = asyncio.run(
        ad.validate_chat(client, entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.passed is False
    assert result.status == db_module.STATUS_TOO_FEW_MEMBERS
    assert "unknown" in result.reason


def test_validate_chat_skips_full_lookup_when_count_known() -> None:
    """A known ``participants_count`` costs no extra full-chat request."""
    entity = make_entity(chat_id=3203, username="known_count", members=1500)
    client = FakeClient(histories={3203: healthy_history()})

    result = asyncio.run(
        ad.validate_chat(client, entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.passed is True
    assert client.full_channel_requests == []
    assert client.full_chat_requests == []


def test_validate_chat_handles_flood_wait_during_full_lookup() -> None:
    """A flood wait while resolving full info is a safe inactive verdict."""
    entity = make_entity(chat_id=3204, username="flooded_chat", members=None)
    client = FakeClient(
        histories={3204: healthy_history()},
        full_channel_effect=FloodWaitError(None, 30),
    )

    result = asyncio.run(
        ad.validate_chat(client, entity, config=TEST_CONFIG, now=REFERENCE)
    )

    assert result.passed is False
    assert result.status == db_module.STATUS_INACTIVE
    assert "30" in result.reason


def test_collect_evaluations_resolves_unknown_member_count(
    audit_db: str,
) -> None:
    """The discovery loop persists the resolved count and the eligible verdict."""
    entity = make_entity(chat_id=3300, username="resolved_chat", members=None)
    client = FakeClient(
        entities={"resolved_chat": entity},
        histories={3300: healthy_history()},
        full_channels={3300: full_channel_response(900)},
    )

    async def scenario() -> list[ad.ChatEvaluation]:
        await db_module.init_db(audit_db)
        return await ad.collect_evaluations(
            client,
            ["resolved_chat"],
            config=TEST_CONFIG,
            now=REFERENCE,
            db_path=audit_db,
        )

    evaluations = asyncio.run(scenario())

    assert len(evaluations) == 1
    assert evaluations[0].status == db_module.STATUS_ELIGIBLE
    assert evaluations[0].members_count == 900
    assert client.full_channel_requests == [entity]


# --------------------------------------------------------------------------- #
# Rate limiting & joining
# --------------------------------------------------------------------------- #


def test_daily_limit_stops_subsequent_join_attempts(audit_db: str) -> None:
    """Once the daily limit is recorded, no join request is ever sent."""

    async def scenario() -> tuple[ad.JoinStats, FakeClient]:
        await db_module.init_db(audit_db)
        for index in range(TEST_LIMIT):
            await db_module.record_chat_evaluation(
                chat_id=900 + index,
                username=f"old_chat_{index}",
                title="Old",
                members_count=1000,
                last_message_date=REFERENCE,
                status=db_module.STATUS_JOINED,
                joined=True,
                db_path=audit_db,
            )
        client = FakeClient()
        evaluation = eligible_evaluation(7777, "fresh_chat")
        stats = await ad.join_eligible_chats(
            client,
            [evaluation],
            config=TEST_CONFIG,
            db_path=audit_db,
            sleep=_noop_sleep,
        )
        return stats, client

    stats, client = asyncio.run(scenario())

    assert stats.joined == []
    assert stats.limit_reached is True
    assert stats.joined_today == TEST_LIMIT
    assert client.join_requests == []
    assert len(stats.skipped_due_to_limit) == 1


def test_run_limit_is_capped_by_daily_limit(audit_db: str) -> None:
    """``--limit`` larger than the daily limit still stops at the daily limit."""

    async def scenario() -> tuple[ad.JoinStats, FakeClient]:
        await db_module.init_db(audit_db)
        client = FakeClient()
        evaluations = [
            eligible_evaluation(8000 + index, f"chat_{index}")
            for index in range(3)
        ]
        stats = await ad.join_eligible_chats(
            client,
            evaluations,
            config=TEST_CONFIG,
            join_limit=10,
            db_path=audit_db,
            sleep=_noop_sleep,
        )
        return stats, client

    stats, client = asyncio.run(scenario())

    assert len(stats.joined) == TEST_LIMIT
    assert len(client.join_requests) == TEST_LIMIT
    assert stats.limit_reached is True
    assert len(stats.skipped_due_to_limit) == 1


def test_join_limit_cannot_exceed_daily_limit(audit_db: str) -> None:
    """A ``--limit`` below the daily limit is honoured exactly."""

    async def scenario() -> tuple[ad.JoinStats, FakeClient]:
        await db_module.init_db(audit_db)
        client = FakeClient()
        evaluations = [
            eligible_evaluation(8100 + index, f"chat_{index}")
            for index in range(3)
        ]
        stats = await ad.join_eligible_chats(
            client,
            evaluations,
            config=TEST_CONFIG,
            join_limit=1,
            db_path=audit_db,
            sleep=_noop_sleep,
        )
        return stats, client

    stats, client = asyncio.run(scenario())

    assert len(stats.joined) == 1
    assert len(client.join_requests) == 1
    assert stats.limit_reached is True


def test_dry_run_never_sends_join_requests(audit_db: str) -> None:
    """``--dry-run`` audits and persists the verdict but never joins."""

    async def scenario() -> tuple[ad.JoinStats, FakeClient, int]:
        await db_module.init_db(audit_db)
        client = FakeClient()
        evaluation = eligible_evaluation(8200, "dry_chat")
        stats = await ad.join_eligible_chats(
            client,
            [evaluation],
            config=TEST_CONFIG,
            dry_run=True,
            db_path=audit_db,
            sleep=_noop_sleep,
        )
        joined_today = await db_module.get_daily_joined_count(audit_db)
        return stats, client, joined_today

    stats, client, joined_today = asyncio.run(scenario())

    assert client.join_requests == []
    assert stats.joined == []
    assert stats.dry_run is True
    assert joined_today == 0


def test_already_participant_is_recorded(audit_db: str) -> None:
    """``UserAlreadyParticipantError`` becomes ``already_joined`` in the audit."""

    async def scenario() -> tuple[ad.JoinStats, Optional[str]]:
        await db_module.init_db(audit_db)
        client = FakeClient(join_effect=UserAlreadyParticipantError(None))
        evaluation = eligible_evaluation(8300, "member_chat")
        stats = await ad.join_eligible_chats(
            client,
            [evaluation],
            config=TEST_CONFIG,
            db_path=audit_db,
            sleep=_noop_sleep,
        )
        status = await db_module.get_chat_status(8300, audit_db)
        return stats, status

    stats, status = asyncio.run(scenario())

    assert len(stats.already_joined) == 1
    assert stats.joined == []
    assert status == db_module.STATUS_ALREADY_JOINED


def test_channels_too_much_stops_the_run(audit_db: str) -> None:
    """``ChannelsTooMuchError`` stops joining and is flagged on the stats."""

    async def scenario() -> tuple[ad.JoinStats, FakeClient]:
        await db_module.init_db(audit_db)
        client = FakeClient(join_effect=ChannelsTooMuchError(None))
        evaluations = [
            eligible_evaluation(8400 + index, f"chat_{index}")
            for index in range(2)
        ]
        stats = await ad.join_eligible_chats(
            client,
            evaluations,
            config=TEST_CONFIG,
            db_path=audit_db,
            sleep=_noop_sleep,
        )
        return stats, client

    stats, client = asyncio.run(scenario())

    assert stats.account_limit is True
    assert stats.joined == []
    assert len(client.join_requests) == 1


def test_flood_wait_is_logged_and_stops_the_run(
    audit_db: str, caplog: pytest.LogCaptureFixture
) -> None:
    """``FloodWaitError`` records the reported seconds and ends the session."""

    async def scenario() -> tuple[ad.JoinStats, FakeClient]:
        await db_module.init_db(audit_db)
        client = FakeClient(join_effect=FloodWaitError(None, 120))
        evaluations = [
            eligible_evaluation(8500 + index, f"chat_{index}")
            for index in range(2)
        ]
        with caplog.at_level("WARNING", logger="lead_sniper.auto_discover"):
            stats = await ad.join_eligible_chats(
                client,
                evaluations,
                config=TEST_CONFIG,
                db_path=audit_db,
                sleep=_noop_sleep,
            )
        return stats, client

    stats, client = asyncio.run(scenario())

    assert stats.flood_wait_seconds == 120
    assert stats.joined == []
    assert len(client.join_requests) == 1
    assert "120" in caplog.text


def test_join_delay_stays_in_the_anti_flood_range() -> None:
    """Join jitter is always inside 180-360 seconds."""
    generator = random.Random(1234)
    samples = [ad.join_delay(ad.DiscoveryConfig(), rng=generator) for _ in range(50)]

    assert all(180.0 <= value <= 360.0 for value in samples)


def test_search_delay_stays_in_the_anti_flood_range() -> None:
    """Search jitter is always inside 10-25 seconds."""
    generator = random.Random(4321)
    samples = [
        ad.search_delay(ad.DiscoveryConfig(), rng=generator) for _ in range(50)
    ]

    assert all(10.0 <= value <= 25.0 for value in samples)


def test_search_usernames_applies_jitter_between_queries() -> None:
    """Exactly one jitter is applied between two queries, and dupes collapse."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = FakeClient(
        chats={
            "first query": [
                SimpleNamespace(username="chat_a", title="A"),
                SimpleNamespace(username="chat_a", title="dup"),
                SimpleNamespace(username=None, title="no username"),
            ],
            "second query": [
                SimpleNamespace(username="chat_b", title="B"),
            ],
        }
    )

    found = asyncio.run(
        ad.search_usernames(
            client,
            ["first query", "second query"],
            config=ad.DiscoveryConfig(),
            sleep=fake_sleep,
            rng=random.Random(0),
        )
    )

    assert found == ["chat_a", "chat_b"]
    assert len(sleeps) == 1
    assert 10.0 <= sleeps[0] <= 25.0


def test_search_usernames_skips_failing_query() -> None:
    """A failing query is logged and skipped without aborting the run."""

    class _FailingSearchClient(FakeClient):
        async def __call__(self, request: Any) -> Any:
            if isinstance(request, SearchRequest) and request.q == "broken":
                self.search_queries.append(request.q)
                raise RuntimeError("boom")
            return await super().__call__(request)

    client = _FailingSearchClient(
        chats={"working": [SimpleNamespace(username="good_chat", title="G")]}
    )

    found = asyncio.run(
        ad.search_usernames(
            client,
            ["broken", "working"],
            config=ad.DiscoveryConfig(),
            sleep=_noop_sleep,
            rng=random.Random(0),
        )
    )

    assert found == ["good_chat"]
    assert client.search_queries == ["broken", "working"]


# --------------------------------------------------------------------------- #
# Session orchestration
# --------------------------------------------------------------------------- #


def test_run_session_dry_run_records_verdict_without_joining(
    audit_db: str,
) -> None:
    """A dry-run session audits and persists passes but never joins."""
    entity = make_entity(chat_id=4242, username="sellers", members=900)
    client = FakeClient(
        chats={"seller keyword": [SimpleNamespace(username="sellers", title="S")]},
        entities={"sellers": entity},
        histories={4242: healthy_history()},
    )

    async def scenario() -> tuple[ad.SessionSummary, Optional[str]]:
        summary = await ad.run_session(
            client,
            ["seller keyword"],
            config=TEST_CONFIG,
            dry_run=True,
            db_path=audit_db,
            now=REFERENCE,
            sleep=_noop_sleep,
        )
        status = await db_module.get_chat_status(4242, audit_db)
        return summary, status

    summary, status = asyncio.run(scenario())

    assert client.join_requests == []
    assert summary.discovered == 1
    assert summary.stats.joined == []
    assert status == db_module.STATUS_ELIGIBLE
    assert list(client.search_queries) == ["seller keyword"]


def test_run_session_skips_already_evaluated_chat(audit_db: str) -> None:
    """Chats recorded with a final status are not evaluated twice."""
    entity = make_entity(chat_id=5151, username="joined_chat")
    client = FakeClient(
        chats={"kw": [SimpleNamespace(username="joined_chat", title="J")]},
        entities={"joined_chat": entity},
        histories={5151: healthy_history()},
    )

    async def scenario() -> ad.SessionSummary:
        await db_module.init_db(audit_db)
        await db_module.record_chat_evaluation(
            chat_id=5151,
            username="joined_chat",
            status=db_module.STATUS_JOINED,
            joined=True,
            db_path=audit_db,
        )
        return await ad.run_session(
            client,
            ["kw"],
            config=TEST_CONFIG,
            dry_run=True,
            db_path=audit_db,
            now=REFERENCE,
            sleep=_noop_sleep,
        )

    summary = asyncio.run(scenario())

    assert summary.discovered == 1
    assert summary.evaluations == []


def test_run_session_respects_reached_daily_limit(audit_db: str) -> None:
    """A live session exits gracefully when today's quota is exhausted."""

    async def scenario() -> tuple[ad.SessionSummary, FakeClient]:
        await db_module.init_db(audit_db)
        for index in range(TEST_LIMIT):
            await db_module.record_chat_evaluation(
                chat_id=6000 + index,
                username=f"done_{index}",
                status=db_module.STATUS_JOINED,
                joined=True,
                db_path=audit_db,
            )
        client = FakeClient()
        summary = await ad.run_session(
            client,
            ["kw"],
            config=TEST_CONFIG,
            dry_run=False,
            db_path=audit_db,
            now=REFERENCE,
            sleep=_noop_sleep,
        )
        return summary, client

    summary, client = asyncio.run(scenario())

    assert summary.limit_reached is True
    assert client.search_queries == []
    assert client.join_requests == []


def test_format_summary_reports_counts() -> None:
    """The summary notification mentions joins and the status breakdown."""
    passed = eligible_evaluation(7001, "fresh_chat")
    rejected = ad.evaluate_chat(
        make_entity(chat_id=7002, username="dead_chat"),
        [make_message(5, sender_id=1)],
        config=TEST_CONFIG,
        now=REFERENCE,
    )
    summary = ad.SessionSummary(
        keywords=3,
        discovered=2,
        evaluations=[passed, rejected],
        stats=ad.JoinStats(joined=[passed], joined_today=1),
    )

    text = ad.format_summary(summary)

    assert "Новых джойнов: 1" in text
    assert "Ключевых слов: 3" in text
    assert "inactive=1" in text


def test_send_summary_is_skipped_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without bot credentials the summary is skipped and never raises."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALERT_CHAT_ID", raising=False)

    assert asyncio.run(ad.send_summary(ad.SessionSummary())) is False


# --------------------------------------------------------------------------- #
# Database audit helpers
# --------------------------------------------------------------------------- #


def test_record_and_is_chat_evaluated(audit_db: str) -> None:
    """A recorded evaluation is visible through both audit helpers."""

    async def scenario() -> tuple[bool, bool, Optional[str], int]:
        await db_module.init_db(audit_db)
        before = await db_module.is_chat_evaluated(101, audit_db)
        await db_module.record_chat_evaluation(
            chat_id=101,
            username="audit_chat",
            title="Audit",
            members_count=321,
            last_message_date=REFERENCE,
            status=db_module.STATUS_INACTIVE,
            db_path=audit_db,
        )
        after = await db_module.is_chat_evaluated(101, audit_db)
        status = await db_module.get_chat_status(101, audit_db)
        joined_today = await db_module.get_daily_joined_count(audit_db)
        return before, after, status, joined_today

    before, after, status, joined_today = asyncio.run(scenario())

    assert before is False
    assert after is True
    assert status == db_module.STATUS_INACTIVE
    assert joined_today == 0


def test_daily_count_tracks_only_joins(audit_db: str) -> None:
    """Only rows actually joined today are counted."""

    async def scenario() -> int:
        await db_module.init_db(audit_db)
        await db_module.record_chat_evaluation(
            chat_id=201,
            username="joined_one",
            status=db_module.STATUS_JOINED,
            joined=True,
            db_path=audit_db,
        )
        await db_module.record_chat_evaluation(
            chat_id=202,
            username="joined_two",
            status=db_module.STATUS_JOINED,
            joined=True,
            db_path=audit_db,
        )
        await db_module.record_chat_evaluation(
            chat_id=203,
            username="rejected",
            status=db_module.STATUS_INACTIVE,
            db_path=audit_db,
        )
        return await db_module.get_daily_joined_count(audit_db)

    assert asyncio.run(scenario()) == 2


def test_reevaluation_preserves_joined_at(audit_db: str) -> None:
    """Re-evaluating a joined chat must not erase its ``joined_at`` stamp."""

    async def scenario() -> tuple[int, Optional[str]]:
        await db_module.init_db(audit_db)
        await db_module.record_chat_evaluation(
            chat_id=301,
            username="joined_chat",
            status=db_module.STATUS_JOINED,
            joined=True,
            db_path=audit_db,
        )
        await db_module.record_chat_evaluation(
            chat_id=301,
            username="joined_chat",
            status=db_module.STATUS_INACTIVE,
            db_path=audit_db,
        )
        return (
            await db_module.get_daily_joined_count(audit_db),
            await db_module.get_chat_status(301, audit_db),
        )

    count, status = asyncio.run(scenario())

    assert count == 1
    assert status == db_module.STATUS_INACTIVE


def test_record_chat_evaluation_rejects_unknown_status(audit_db: str) -> None:
    """An unrecognised status is a programming error, not silent data loss."""
    with pytest.raises(ValueError):
        asyncio.run(
            db_module.record_chat_evaluation(
                chat_id=401,
                status="definitely_not_a_status",
                db_path=audit_db,
            )
        )


def test_is_chat_evaluated_requires_int(audit_db: str) -> None:
    """The helper enforces its integer contract."""
    with pytest.raises(TypeError):
        asyncio.run(db_module.is_chat_evaluated("401", audit_db))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Configuration & CLI
# --------------------------------------------------------------------------- #


def test_load_config_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thresholds come from the DISCOVERY_* environment variables."""
    monkeypatch.setenv("DISCOVERY_MIN_MEMBERS", "80")
    monkeypatch.setenv("DISCOVERY_DAILY_JOIN_LIMIT", "3")

    config = ad.load_config()

    assert config.min_members == 80
    assert config.daily_join_limit == 3


def test_load_config_uses_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no DISCOVERY_* variables the documented defaults apply."""
    for name in (
        "DISCOVERY_MIN_MEMBERS",
        "DISCOVERY_DAILY_JOIN_LIMIT",
        "DISCOVERY_MAX_AGE_HOURS",
        "DISCOVERY_MIN_MESSAGES_7D",
        "DISCOVERY_HISTORY_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = ad.load_config()

    assert config.min_members == 50
    assert config.daily_join_limit == 5
    assert config.max_age_hours == 48.0
    assert config.min_messages_7d == 5
    assert config.history_limit == 10


def test_load_config_falls_back_on_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric value is logged and replaced by the default."""
    monkeypatch.setenv("DISCOVERY_MIN_MEMBERS", "many")

    assert ad.load_config().min_members == 50


def test_parse_args_defaults() -> None:
    """Discovery runs live by default, with a daily-limit-sized join cap."""
    args = ad.parse_args([])

    assert args.dry_run is False
    assert args.limit is None
    assert args.keyword is None
    assert args.db == db_module.DEFAULT_DB_PATH


def test_parse_args_reads_flags() -> None:
    """``--dry-run``, ``--limit``, repeated ``--keyword`` and ``--db`` parse."""
    args = ad.parse_args(
        [
            "--dry-run",
            "--limit",
            "2",
            "--keyword",
            "wb чат",
            "--keyword",
            "ozon селлеры",
            "--db",
            "custom.db",
        ]
    )

    assert args.dry_run is True
    assert args.limit == 2
    assert args.keyword == ["wb чат", "ozon селлеры"]
    assert args.db == "custom.db"


def test_parse_args_rejects_negative_limit() -> None:
    """A negative ``--limit`` is a usage error."""
    with pytest.raises(SystemExit):
        ad.parse_args(["--limit", "-1"])


def test_resolve_keywords_prefers_cli_values() -> None:
    """Repeated keywords are stripped, ordered and deduplicated."""
    assert ad.resolve_keywords([" wb чат ", "ozon", "wb чат", "  "]) == [
        "wb чат",
        "ozon",
    ]


def test_resolve_keywords_falls_back_to_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--keyword`` the keywords file is used."""
    keywords_file = tmp_path / "keywords.txt"
    keywords_file.write_text("# comment\nfrom_file\n", encoding="utf-8")
    monkeypatch.setattr(join_chats, "KEYWORDS_FILE", keywords_file)

    assert ad.resolve_keywords(None) == ["from_file"]
    assert ad.resolve_keywords(["   "]) == ["from_file"]
