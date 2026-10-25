"""Sprint 4 — verification of the async Telegram alert notifier.

Run with::

    .venv\\Scripts\\python.exe test_notifier.py

The suite has two parts:

* **Offline (always runs).** ``httpx.MockTransport`` feeds canned responses and
  exceptions into :func:`notifier.send_lead_alert`, covering the success path,
  429 rate-limit retries, permanent 4xx failures, 5xx retries and network
  timeouts. :func:`notifier.format_alert` is checked against long texts,
  missing links, missing fields and HTML-escaping edge cases. No network access
  and no bot token are required.
* **Live (optional).** When both ``TELEGRAM_BOT_TOKEN`` and
  ``TELEGRAM_ALERT_CHAT_ID`` are present in ``.env`` (or the process
  environment), one real alert is dispatched to prove the credentials work end
  to end.

Exit code ``0`` means every executed check passed; ``1`` means at least one
failed.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Final, Iterator
from unittest import mock

import httpx

import notifier
from notifier import (
    DEFAULT_CATEGORY,
    MAX_MESSAGE_LENGTH,
    TelegramConfig,
    format_alert,
    load_telegram_config,
    read_telegram_config,
    send_lead_alert,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

LEAD: Final[dict[str, Any]] = {
    "is_lead": True,
    "category": "Карго / Логистика",
    "summary": "Поиск карго из Китая в Москву",
    "confidence": 9,
}

ORIGINAL: Final[str] = (
    "Ищу карго из Китая в Москву, кто занимается доставкой под ключ? "
    "Нужен надёжный подрядчик."
)

SOURCE: Final[str] = "Карго Чат Москва"

LINK: Final[str] = "https://t.me/cargo_chat/12345"

TEST_CONFIG: Final[TelegramConfig] = TelegramConfig(
    bot_token="123456:TEST-TOKEN",
    chat_id="-1001234567890",
)


@contextlib.contextmanager
def _tg_env(values: dict[str, str] | None = None) -> Iterator[None]:
    """Temporarily replace the ``TELEGRAM_*`` environment variables."""
    keys = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALERT_CHAT_ID")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    if values:
        os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """Build an ``AsyncClient`` backed by a mock transport."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


class Reporter:
    """Collects check results and prints them in a readable, aligned format."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failures: list[str] = []

    def section(self, title: str) -> None:
        print()
        print("-" * 76)
        print(title)
        print("-" * 76)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        """Record a single expectation and print its outcome immediately."""
        status = "[PASS]" if condition else "[FAIL]"
        line = f"  {status} {name}"
        if detail:
            line += f"  ->  {detail}"
        print(line)
        if condition:
            self.passed += 1
        else:
            self.failures.append(name)

    def summary(self) -> None:
        total = self.passed + len(self.failures)
        print()
        print("=" * 76)
        print(f" RESULTS: {self.passed}/{total} checks passed")
        if self.failures:
            print(f" FAILED CHECKS ({len(self.failures)}):")
            for name in self.failures:
                print(f"   - {name}")
        else:
            print(" All checks passed.")
        print("=" * 76)


# --------------------------------------------------------------------------- #
# Test suites
# --------------------------------------------------------------------------- #


async def test_configuration(reporter: Reporter) -> None:
    """Defaults must be clean and chat ids must be validated."""
    reporter.section("1. Configuration loading & validation")

    with _tg_env():
        cfg = await load_telegram_config(env_file=Path("definitely-missing.env"))

    reporter.check(
        "missing token leaves the notifier disabled",
        cfg.bot_token == "" and cfg.is_configured is False,
        f"bot_token={cfg.bot_token!r}",
    )
    reporter.check("missing chat id is empty", cfg.chat_id == "", repr(cfg.chat_id))

    with _tg_env(
        {
            "TELEGRAM_BOT_TOKEN": "  123:ABC  ",
            "TELEGRAM_ALERT_CHAT_ID": "  -1001234567890  ",
        }
    ):
        numeric = read_telegram_config()

    reporter.check(
        "token and numeric chat id are stripped and accepted",
        numeric
        == TelegramConfig(bot_token="123:ABC", chat_id="-1001234567890"),
        f"{numeric.bot_token!r} / {numeric.chat_id!r}",
    )
    reporter.check(
        "numeric config is considered configured",
        numeric.is_configured is True,
    )

    with _tg_env(
        {
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_ALERT_CHAT_ID": "@my_leads",
        }
    ):
        username = read_telegram_config()

    reporter.check(
        "@username chat id is accepted",
        username.chat_id == "@my_leads" and username.is_configured is True,
        repr(username.chat_id),
    )

    with _tg_env(
        {
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_ALERT_CHAT_ID": "not a chat",
        }
    ):
        invalid = read_telegram_config()

    reporter.check(
        "invalid chat id disables the notifier",
        invalid.chat_id == "" and invalid.is_configured is False,
        repr(invalid.chat_id),
    )

    with _tg_env({"TELEGRAM_BOT_TOKEN": "123:ABC"}):
        no_chat = read_telegram_config()

    reporter.check(
        "token without chat id is not configured",
        no_chat.is_configured is False,
    )


async def test_format_alert(reporter: Reporter) -> None:
    """The rendered body must be readable, escaped and length-capped."""
    reporter.section("2. Alert formatting (content, escaping, edge cases)")

    text = format_alert(LEAD, ORIGINAL, SOURCE, LINK)

    reporter.check(
        "header contains the category",
        "Карго / Логистика" in text,
    )
    reporter.check(
        "header contains the confidence score",
        "Рейтинг: 9/10" in text,
    )
    reporter.check(
        "summary is rendered in bold",
        "<b>Поиск карго из Китая в Москву</b>" in text,
    )
    reporter.check("source chat is present", SOURCE in text)
    reporter.check(
        "message link is rendered as an anchor",
        f'href="{LINK}"' in text and "Открыть сообщение" in text,
    )
    reporter.check(
        "original text is quoted",
        "<blockquote>" in text and "Ищу карго" in text,
    )

    no_link = format_alert(LEAD, ORIGINAL, SOURCE)
    reporter.check(
        "missing link omits the link line",
        "Открыть сообщение" not in no_link and "href=" not in no_link,
    )

    invalid_link = format_alert(LEAD, ORIGINAL, SOURCE, "javascript:alert(1)")
    reporter.check(
        "non-http link is dropped",
        "href=" not in invalid_link,
    )

    escaped = format_alert(
        {
            "category": "<b>Карго</b> & Логистика",
            "summary": "A < B & C",
            "confidence": 7,
        },
        '<script>alert("x")</script> & co',
        "Чат <VIP>",
        "https://t.me/x?a=1&b=2",
    )
    reporter.check(
        "category is HTML-escaped",
        html.escape("<b>Карго</b> & Логистика", quote=False) in escaped,
    )
    reporter.check(
        "summary is HTML-escaped",
        html.escape("A < B & C", quote=False) in escaped,
    )
    reporter.check(
        "snippet is HTML-escaped",
        html.escape('<script>alert("x")</script> & co', quote=False) in escaped
        and "<script>" not in escaped,
    )
    reporter.check(
        "source is HTML-escaped",
        html.escape("Чат <VIP>", quote=False) in escaped,
    )
    reporter.check(
        "link attribute is HTML-escaped",
        html.escape("https://t.me/x?a=1&b=2", quote=True) in escaped,
    )

    clamped_high = format_alert({**LEAD, "confidence": 42}, ORIGINAL, SOURCE)
    clamped_low = format_alert({**LEAD, "confidence": -3}, ORIGINAL, SOURCE)
    reporter.check(
        "confidence above 10 is clamped",
        "Рейтинг: 10/10" in clamped_high,
    )
    reporter.check(
        "confidence below 1 is clamped",
        "Рейтинг: 1/10" in clamped_low,
    )

    minimal = format_alert({}, "", "")
    reporter.check(
        "missing category falls back to the default",
        DEFAULT_CATEGORY in minimal,
    )
    reporter.check(
        "missing confidence omits the rating",
        "Рейтинг" not in minimal,
    )
    reporter.check(
        "missing source falls back to a placeholder",
        "неизвестный чат" in minimal,
    )

    fallback_summary = format_alert({"category": "Другое"}, ORIGINAL, SOURCE)
    reporter.check(
        "missing summary falls back to the original text",
        "Ищу карго" in fallback_summary,
    )

    long_text = "Нужен карго из Китая в Москву. " * 1000
    long_alert = format_alert(LEAD, long_text, SOURCE)
    reporter.check(
        "long snippet is truncated with an ellipsis",
        "…" in long_alert,
    )
    reporter.check(
        "default snippet stays within Telegram's limit",
        len(long_alert) <= MAX_MESSAGE_LENGTH,
        str(len(long_alert)),
    )

    with mock.patch.object(notifier, "MAX_SNIPPET_CHARS", 10000):
        huge = format_alert(LEAD, long_text, SOURCE)
    reporter.check(
        "message body never exceeds Telegram's 4096-char limit",
        len(huge) <= MAX_MESSAGE_LENGTH,
        str(len(huge)),
    )


async def test_delivery_success(reporter: Reporter) -> None:
    """A 200 response must return ``True`` and send the expected payload."""
    reporter.section("3. Delivery — success path")

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 1}},
            request=request,
        )

    async with _client(handler) as client:
        ok = await send_lead_alert(
            LEAD,
            ORIGINAL,
            SOURCE,
            LINK,
            config=TEST_CONFIG,
            client=client,
            max_retries=0,
        )

    reporter.check("successful send returns True", ok is True)
    reporter.check(
        "exactly one request was made",
        len(requests) == 1,
        str(len(requests)),
    )

    if requests:
        request = requests[0]
        reporter.check(
            "endpoint includes the bot token",
            request.url.path == "/bot123456:TEST-TOKEN/sendMessage",
            str(request.url),
        )
        payload = json.loads(request.content)
        reporter.check(
            "payload targets the configured chat",
            payload.get("chat_id") == TEST_CONFIG.chat_id,
            repr(payload.get("chat_id")),
        )
        reporter.check(
            "payload uses HTML parse mode",
            payload.get("parse_mode") == "HTML",
            repr(payload.get("parse_mode")),
        )
        reporter.check(
            "payload contains the formatted alert",
            "Карго / Логистика" in payload.get("text", ""),
        )
        reporter.check(
            "link previews are disabled",
            payload.get("disable_web_page_preview") is True,
        )


async def test_rate_limit_retry(reporter: Reporter) -> None:
    """A 429 must be retried and then succeed."""
    reporter.section("4. Delivery — 429 rate limit retry")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                429,
                json={
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests: retry after 0",
                    "parameters": {"retry_after": 0},
                },
                headers={"Retry-After": "0"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 2}},
            request=request,
        )

    with mock.patch.object(notifier, "RETRY_BASE_DELAY", 0.0), mock.patch.object(
        notifier, "MAX_RETRY_DELAY", 0.0
    ):
        async with _client(handler) as client:
            ok = await send_lead_alert(
                LEAD,
                ORIGINAL,
                SOURCE,
                config=TEST_CONFIG,
                client=client,
                max_retries=1,
            )

    reporter.check(
        "429 is retried and then succeeds",
        ok is True and calls["count"] == 2,
        f"ok={ok}, calls={calls['count']}",
    )


async def test_bad_request(reporter: Reporter) -> None:
    """A permanent 4xx must fail immediately without retrying."""
    reporter.section("5. Delivery — permanent 4xx failure")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            400,
            json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: chat not found",
            },
            request=request,
        )

    async with _client(handler) as client:
        ok = await send_lead_alert(
            LEAD,
            ORIGINAL,
            SOURCE,
            config=TEST_CONFIG,
            client=client,
            max_retries=2,
        )

    reporter.check("400 returns False", ok is False)
    reporter.check(
        "400 is not retried",
        calls["count"] == 1,
        str(calls["count"]),
    )


async def test_network_timeout(reporter: Reporter) -> None:
    """A network timeout must be retried and then reported as failure."""
    reporter.section("6. Delivery — network timeout")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectTimeout("simulated timeout", request=request)

    with mock.patch.object(notifier, "RETRY_BASE_DELAY", 0.0), mock.patch.object(
        notifier, "MAX_RETRY_DELAY", 0.0
    ):
        async with _client(handler) as client:
            ok = await send_lead_alert(
                LEAD,
                ORIGINAL,
                SOURCE,
                config=TEST_CONFIG,
                client=client,
                max_retries=1,
            )

    reporter.check("timeout returns False", ok is False)
    reporter.check(
        "timeout is retried max_retries + 1 times",
        calls["count"] == 2,
        str(calls["count"]),
    )


async def test_server_error_retry(reporter: Reporter) -> None:
    """A 5xx must be retried and then succeed."""
    reporter.section("7. Delivery — 5xx retry")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                502,
                json={"ok": False, "error_code": 502},
                request=request,
            )
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 3}},
            request=request,
        )

    with mock.patch.object(notifier, "RETRY_BASE_DELAY", 0.0), mock.patch.object(
        notifier, "MAX_RETRY_DELAY", 0.0
    ):
        async with _client(handler) as client:
            ok = await send_lead_alert(
                LEAD,
                ORIGINAL,
                SOURCE,
                config=TEST_CONFIG,
                client=client,
                max_retries=1,
            )

    reporter.check(
        "5xx is retried and then succeeds",
        ok is True and calls["count"] == 2,
        f"ok={ok}, calls={calls['count']}",
    )


async def test_unconfigured(reporter: Reporter) -> None:
    """Missing credentials must short-circuit without any request."""
    reporter.section("8. Delivery — configuration guard")

    called = {"value": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["value"] = True
        return httpx.Response(200, json={"ok": True}, request=request)

    async with _client(handler) as client:
        ok = await send_lead_alert(
            LEAD,
            ORIGINAL,
            SOURCE,
            config=TelegramConfig(),
            client=client,
            max_retries=0,
        )

    reporter.check("unconfigured notifier returns False", ok is False)
    reporter.check(
        "unconfigured notifier makes no request",
        called["value"] is False,
    )


async def test_live_dispatch(reporter: Reporter) -> None:
    """Optionally send a real alert when the bot credentials are configured."""
    reporter.section("9. Live dispatch (optional)")

    config = await load_telegram_config()
    if not config.is_configured:
        print(
            "  [SKIP] TELEGRAM_BOT_TOKEN / TELEGRAM_ALERT_CHAT_ID are not set "
            "in .env or the environment — offline checks only."
        )
        return

    print(f"  [INFO] Sending a real test alert to chat {config.chat_id}.")
    ok = await send_lead_alert(
        LEAD,
        ORIGINAL,
        "Sprint 4 test",
        None,
        config=config,
        max_retries=1,
    )
    reporter.check("live dispatch returns True", ok is True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run_suite() -> int:
    """Execute every test suite and return a process exit code."""
    reporter = Reporter()

    print("=" * 76)
    print(" Sprint 4 — async Telegram alert notifier verification")
    print("=" * 76)

    await test_configuration(reporter)
    await test_format_alert(reporter)
    await test_delivery_success(reporter)
    await test_rate_limit_retry(reporter)
    await test_bad_request(reporter)
    await test_network_timeout(reporter)
    await test_server_error_retry(reporter)
    await test_unconfigured(reporter)
    await test_live_dispatch(reporter)

    reporter.summary()
    assert not reporter.failures, (
        f"{len(reporter.failures)} check(s) failed: "
        + ", ".join(reporter.failures)
    )
    print()
    print("[SUCCESS] Sprint 4 verification completed with 100% checks passing.")
    return 0


def main() -> int:
    """Synchronous wrapper so the script can be executed directly."""
    try:
        return asyncio.run(run_suite())
    except AssertionError as exc:
        print()
        print(f"[FAILURE] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
