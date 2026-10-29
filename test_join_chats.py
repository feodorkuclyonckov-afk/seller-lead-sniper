"""Tests for :mod:`join_chats` proxy and environment resolution.

These mirror the direct-connection contract already covered for
``auth_test.py``: an unset, blank or loopback ``PROXY_HOST`` must always mean
"connect directly to Telegram", and the project ``.env`` must win over a stale
proxy address exported by the shell.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

import join_chats


# --------------------------------------------------------------------------- #
# proxy configuration
# --------------------------------------------------------------------------- #


def test_build_proxy_returns_none_when_host_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ``PROXY_HOST`` means a direct connection."""
    monkeypatch.delenv("PROXY_HOST", raising=False)
    assert join_chats.build_proxy() is None


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_build_proxy_returns_none_for_blank_host(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Empty or whitespace-only ``PROXY_HOST`` means a direct connection."""
    monkeypatch.setenv("PROXY_HOST", value)
    assert join_chats.build_proxy() is None


@pytest.mark.parametrize("value", ["127.0.0.1", "localhost", "LOCALHOST"])
def test_build_proxy_returns_none_for_loopback(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Loopback hosts are treated as "no proxy"."""
    monkeypatch.setenv("PROXY_HOST", value)
    assert join_chats.build_proxy() is None


def test_build_proxy_logs_direct_connection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The direct-connection path emits the documented info message."""
    monkeypatch.setenv("PROXY_HOST", "   ")
    with caplog.at_level("INFO", logger="lead_sniper.join_chats"):
        assert join_chats.build_proxy() is None
    assert "No proxy configured — connecting directly to Telegram." in caplog.text


def test_build_proxy_builds_dict_for_real_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real host produces a python-socks style proxy dict."""
    monkeypatch.setenv("PROXY_HOST", " 196.18.15.173 ")
    monkeypatch.setenv("PROXY_PORT", "8000")
    monkeypatch.setenv("PROXY_TYPE", "http")
    monkeypatch.delenv("PROXY_USER", raising=False)
    monkeypatch.delenv("PROXY_PASS", raising=False)
    # The proxy dict is built before Telethon imports python-socks; stub the
    # optional dependency so the test does not depend on the install extras.
    monkeypatch.setattr(join_chats, "python_socks", object())

    assert join_chats.build_proxy() == {
        "proxy_type": "http",
        "addr": "196.18.15.173",
        "port": 8000,
        "rdns": False,
    }


def test_load_environment_overrides_stale_proxy_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty ``PROXY_HOST`` in ``.env`` must win over a stale process value.

    This reproduces the reported bug: the shell still exported the old proxy
    address, and ``load_dotenv(override=False)`` left it in place, so the
    client silently kept using the proxy even though ``.env`` said direct.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("PROXY_HOST=\n", encoding="utf-8")
    monkeypatch.setattr(join_chats, "ENV_FILE", env_file)
    monkeypatch.setenv("PROXY_HOST", "196.18.15.173")

    join_chats.load_environment()

    assert os.getenv("PROXY_HOST") == ""
    assert join_chats.build_proxy() is None


# --------------------------------------------------------------------------- #
# target loading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("wb_chat_sellers", "wb_chat_sellers"),
        ("  wb_chat_sellers  ", "wb_chat_sellers"),
        ("@wb_chat_sellers", "wb_chat_sellers"),
        ("https://t.me/wb_chat_sellers", "wb_chat_sellers"),
        ("http://t.me/wb_chat_sellers", "wb_chat_sellers"),
        ("https://telegram.me/wb_chat_sellers", "wb_chat_sellers"),
        ("t.me/wb_chat_sellers", "wb_chat_sellers"),
        ("https://t.me/wb_chat_sellers/123", "wb_chat_sellers"),
        ("https://t.me/wb_chat_sellers?start=1", "wb_chat_sellers"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_target_strips_prefixes(raw: str, expected: str) -> None:
    """Usernames and links are normalised to a bare username."""
    assert join_chats.normalize_target(raw) == expected


def test_load_targets_falls_back_to_builtin_when_file_missing(
    tmp_path: Path,
) -> None:
    """A missing ``targets.txt`` keeps the built-in pool working."""
    missing = tmp_path / "targets.txt"
    assert join_chats.load_targets(missing) == list(join_chats.TARGET_CHATS)


def test_load_targets_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    """Empty lines and ``#`` comments are skipped."""
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text(
        "# header comment\n"
        "\n"
        "wb_chat_sellers\n"
        "   \n"
        "  # indented comment\n"
        "fulfillment_russia\n",
        encoding="utf-8",
    )

    assert join_chats.load_targets(targets_file) == [
        "wb_chat_sellers",
        "fulfillment_russia",
    ]


def test_load_targets_strips_prefixes_and_deduplicates(tmp_path: Path) -> None:
    """Links/``@`` prefixes are stripped and duplicates keep first-seen order."""
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text(
        "@wb_chat_sellers\n"
        "https://t.me/wb_chat_sellers\n"
        "t.me/fulfillment_russia\n"
        "fulfillment_russia\n",
        encoding="utf-8",
    )

    assert join_chats.load_targets(targets_file) == [
        "wb_chat_sellers",
        "fulfillment_russia",
    ]


def test_load_targets_logs_loaded_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The number of loaded targets is logged on start."""
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text("wb_chat_sellers\nfulfillment_russia\n", encoding="utf-8")

    with caplog.at_level("INFO", logger="lead_sniper.join_chats"):
        join_chats.load_targets(targets_file)

    assert "Loaded 2 target chat(s)" in caplog.text


# --------------------------------------------------------------------------- #
# search query loading
# --------------------------------------------------------------------------- #


def test_load_queries_falls_back_to_builtin_when_file_missing(
    tmp_path: Path,
) -> None:
    """A missing ``keywords.txt`` keeps the built-in niche queries working."""
    missing = tmp_path / "keywords.txt"
    assert join_chats.load_queries(missing) == list(
        join_chats.DEFAULT_SEARCH_QUERIES
    )


def test_load_queries_ignores_blank_lines_comments_and_duplicates(
    tmp_path: Path,
) -> None:
    """Queries are stripped, comments skipped and duplicates collapsed."""
    keywords_file = tmp_path / "keywords.txt"
    keywords_file.write_text(
        "# niche keywords\n"
        "\n"
        "  wildberries карго  \n"
        "ozon фулфилмент\n"
        "wildberries карго\n"
        "   # indented comment\n",
        encoding="utf-8",
    )

    assert join_chats.load_queries(keywords_file) == [
        "wildberries карго",
        "ozon фулфилмент",
    ]


def test_resolve_queries_prefers_cli_queries(tmp_path: Path) -> None:
    """Explicit ``--query`` values win over ``keywords.txt``."""
    keywords_file = tmp_path / "keywords.txt"
    keywords_file.write_text("from file\n", encoding="utf-8")

    assert join_chats.resolve_queries(["  a  ", "b", "a", "  "], keywords_file) == [
        "a",
        "b",
    ]


def test_resolve_queries_falls_back_when_cli_queries_blank(tmp_path: Path) -> None:
    """Blank ``--query`` values fall back to the keywords file."""
    keywords_file = tmp_path / "keywords.txt"
    keywords_file.write_text("from file\n", encoding="utf-8")

    assert join_chats.resolve_queries(["   "], keywords_file) == ["from file"]
    assert join_chats.resolve_queries(None, keywords_file) == ["from file"]


# --------------------------------------------------------------------------- #
# chat filtering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("chat", "expected"),
    [
        # Public megagroup with a username: the ideal target.
        (
            SimpleNamespace(
                username="wb_chat_sellers", megagroup=True, broadcast=False, left=True
            ),
            True,
        ),
        # A megagroup without an explicit ``left`` flag is still accepted.
        (
            SimpleNamespace(username="no_left_flag", megagroup=True, broadcast=False),
            True,
        ),
        # Broadcast-only channel: clients cannot post requests there.
        (
            SimpleNamespace(
                username="wb_news", megagroup=False, broadcast=True, left=True
            ),
            False,
        ),
        # Broadcast channel with comments enabled is explicitly allowed.
        (
            SimpleNamespace(
                username="wb_comments",
                megagroup=False,
                broadcast=True,
                left=True,
                comments_enabled=True,
            ),
            True,
        ),
        # A channel with neither flag is not a discussion group.
        (
            SimpleNamespace(
                username="empty_channel", megagroup=False, broadcast=False, left=True
            ),
            False,
        ),
        # No public username: cannot be joined by link.
        (SimpleNamespace(username=None, megagroup=True, broadcast=False), False),
        (SimpleNamespace(username="   ", megagroup=True, broadcast=False), False),
        # ``left is False`` means the account is already a member: excluded.
        (
            SimpleNamespace(
                username="already_here", megagroup=True, broadcast=False, left=False
            ),
            False,
        ),
        # ``left`` is True means the account is *not* a member (Telethon
        # semantics), so this chat is joinable.
        (
            SimpleNamespace(
                username="left_chat", megagroup=True, broadcast=False, left=True
            ),
            True,
        ),
        # Deactivated chats can never be joined.
        (
            SimpleNamespace(
                username="dead_chat",
                megagroup=True,
                broadcast=False,
                deactivated=True,
            ),
            False,
        ),
        # Basic group (no megagroup/broadcast flags) with a username.
        (SimpleNamespace(username="basic_group", left=True), True),
        # Basic group already joined (left=False) is excluded.
        (SimpleNamespace(username="basic_joined", left=False), False),
    ],
)
def test_is_joinable_chat(chat: object, expected: bool) -> None:
    """Only public discussion groups/supergroups are accepted."""
    assert join_chats._is_joinable_chat(chat) is expected


# --------------------------------------------------------------------------- #
# global search
# --------------------------------------------------------------------------- #


class _FakeSearchClient:
    """Minimal stand-in for ``TelegramClient``'s callable surface."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.queries: list[str] = []

    async def __call__(self, request: Any) -> Any:
        self.queries.append(request.q)
        response = self.responses[request.q]
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(chats=response)


def _channel(
    username: Optional[str],
    *,
    megagroup: bool = True,
    broadcast: bool = False,
    left: bool = True,
    title: str = "t",
) -> SimpleNamespace:
    return SimpleNamespace(
        username=username,
        megagroup=megagroup,
        broadcast=broadcast,
        left=left,
        title=title,
    )


def test_search_target_chats_filters_and_deduplicates() -> None:
    """Only joinable chats are kept; usernames are deduplicated in order."""
    client = _FakeSearchClient(
        {
            "query one": [
                _channel("first", title="First"),
                _channel(None, title="No username"),
                _channel("news", megagroup=False, broadcast=True, title="News"),
                _channel("first", title="Duplicate"),
                _channel("second", title="Second"),
            ],
            "query two": [
                _channel("third", title="Third"),
                _channel("second", title="Duplicate again"),
            ],
        }
    )

    found = asyncio.run(
        join_chats.search_target_chats(client, ["query one", "query two"], limit=20)
    )

    assert found == ["first", "second", "third"]
    assert client.queries == ["query one", "query two"]


def test_search_target_chats_skips_failing_query() -> None:
    """A failing query is logged and skipped without aborting the run."""
    client = _FakeSearchClient(
        {
            "broken": RuntimeError("boom"),
            "working": [_channel("good_chat", title="Good")],
        }
    )

    found = asyncio.run(
        join_chats.search_target_chats(client, ["broken", "working"])
    )

    assert found == ["good_chat"]
    assert client.queries == ["broken", "working"]


def test_search_target_chats_ignores_blank_queries() -> None:
    """Blank query strings are never sent to Telegram."""
    client = _FakeSearchClient({})

    found = asyncio.run(join_chats.search_target_chats(client, ["  ", ""]))

    assert found == []
    assert client.queries == []


# --------------------------------------------------------------------------- #
# discovery persistence
# --------------------------------------------------------------------------- #


def test_append_targets_adds_only_new_entries(tmp_path: Path) -> None:
    """New usernames are appended; existing entries are preserved verbatim."""
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text(
        "# curated\nwb_chat_sellers\nfulfillment_russia\n",
        encoding="utf-8",
    )

    added = join_chats.append_targets(
        ["fulfillment_russia", "@new_chat", "https://t.me/other_chat"],
        targets_file,
    )

    assert added == ["new_chat", "other_chat"]
    text = targets_file.read_text(encoding="utf-8")
    assert "# curated" in text
    assert "wb_chat_sellers" in text
    assert "new_chat" in text
    assert "other_chat" in text
    assert "# Discovered via join_chats.py --discover" in text
    # Loading the merged file yields every target exactly once.
    assert join_chats.load_targets(targets_file) == [
        "wb_chat_sellers",
        "fulfillment_russia",
        "new_chat",
        "other_chat",
    ]


def test_append_targets_is_idempotent(tmp_path: Path) -> None:
    """Re-appending known chats writes nothing and reports no additions."""
    targets_file = tmp_path / "targets.txt"
    targets_file.write_text("existing_chat\n", encoding="utf-8")

    assert join_chats.append_targets(["existing_chat"], targets_file) == []
    assert targets_file.read_text(encoding="utf-8") == "existing_chat\n"


# --------------------------------------------------------------------------- #
# CLI flags
# --------------------------------------------------------------------------- #


def test_parse_args_defaults_disable_discovery() -> None:
    """Discovery is opt-in and no queries are configured by default."""
    args = join_chats.parse_args([])

    assert args.discover is False
    assert args.query is None
    assert args.limit == join_chats.DEFAULT_LIMIT


@pytest.mark.parametrize("flag", ["--discover", "--auto-search"])
def test_parse_args_enable_discovery(flag: str) -> None:
    """Both flag spellings enable automatic discovery."""
    assert join_chats.parse_args([flag]).discover is True


def test_parse_args_collects_repeated_queries() -> None:
    """``--query`` is repeatable and preserves order."""
    args = join_chats.parse_args(
        ["--discover", "--query", "wildberries карго", "--query", "ozon фулфилмент"]
    )

    assert args.discover is True
    assert args.query == ["wildberries карго", "ozon фулфилмент"]
