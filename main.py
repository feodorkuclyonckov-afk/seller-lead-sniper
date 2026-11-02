"""Sprint 5 — CLI entry point and Telethon runner for the Lead Sniper.

Modes
-----
``--demo``
    Interactive, offline-friendly simulation: a handful of marketplace seller
    messages is fed through :func:`pipeline.process_raw_message` with a
    configurable delay between them. Every stage is logged, and any message that
    survives all four steps dispatches a *real* Telegram alert to the configured
    channel (configure ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_ALERT_CHAT_ID`` in
    ``.env``).

``--live``
    Starts a Telethon client, listens for incoming messages on the watched
    chats and pushes each one through the same pipeline. Chat targets may be
    supplied via ``--chats`` or the ``TG_MONITOR_CHATS`` environment variable
    (comma-separated usernames/ids). With no targets the client listens to
    every chat the account receives messages from.

Shutdown
--------
``SIGINT`` (Ctrl+C) and ``SIGTERM`` set an :class:`asyncio.Event`; the live
listener then disconnects cleanly and closes the LLM HTTP client. The handler
falls back to :func:`signal.signal` on platforms (Windows) where
``loop.add_signal_handler`` is unavailable.

Common flags
------------
``--db PATH``       Deduplication SQLite cache (default ``leads_cache.db``).
``--session NAME``  Telethon session name (default ``TG_SESSION_NAME``, falling
                    back to ``session_auth_test``).
``--delay SECONDS`` Delay between demo messages (default ``2.0``).
``--chats LIST``    Comma-separated chat usernames/ids for ``--live``.
``--log-level``     Logging verbosity (default ``INFO``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

import classifier
import db as db_module
import pipeline
from db import DEFAULT_DB_PATH

LOGGER = logging.getLogger("lead_sniper.main")

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

#: Fallback session name, kept in sync with :mod:`auth_test`.
DEFAULT_SESSION: str = "session_auth_test"
DEFAULT_DELAY: float = 2.0
DEFAULT_LOG_LEVEL: str = "INFO"

ENV_FILE: Path = Path(__file__).resolve().with_name(".env")

#: Environment variable holding a comma-separated list of chats for ``--live``.
MONITOR_CHATS_ENV: str = "TG_MONITOR_CHATS"

EXIT_OK: int = 0
EXIT_CONFIG_ERROR: int = 1
EXIT_RUNTIME_ERROR: int = 2

#: Sample marketplace seller messages fed through the pipeline in demo mode.
#: Each tuple is ``(text, source_chat, msg_link)``. The first four are real
#: pain-point leads; the last two are locally filtered noise.
DEMO_MESSAGES: tuple[tuple[str, str, str | None], ...] = (
    (
        "Включили принудительную автоакцию на весь ассортимент, цены срезали "
        "ниже себестоимости, скидки рисуют сами, маржа уже в минус. "
        "Кто сталкивался, как отключить?",
        "WB Селлеры Чат",
        "https://t.me/wb_sellers_chat/12345",
    ),
    (
        "С 11 числа WB поднял коэффициент склада до х5, логистика и хранение "
        "пересчитали задним числом, доплата выросла в три раза. "
        "Кто оспаривал такие начисления?",
        "WB Логистика Чат",
        "https://t.me/wb_logistics_chat/777",
    ),
    (
        "Прилетел штраф за перемер габаритов: удержали 40 тысяч из отчёта "
        "реализации, хотя упаковка не менялась. Как доказать, что замер "
        "неверный?",
        "WB Штрафы Чат",
        "https://t.me/wb_fines_chat/4321",
    ),
    (
        "Из-за рекламы и скидок пошёл полный слив маржи, товар продаётся "
        "в убыток, уже кассовый разрыв на закупку следующей партии. "
        "Что делать?",
        "Ozon Селлеры Чат",
        "https://t.me/ozon_sellers_chat/555",
    ),
    (
        "Предлагаю услуги по накрутке отзывов и продвижению карточек WB, "
        "пишите в личку, выведу в топ за месяц.",
        "Маркетплейс Услуги",
        None,
    ),
    (
        "Всем привет, как у вас дела?",
        "Болталка",
        None,
    ),
)


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


def configure_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure the single, structured log format used by the whole process."""
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Telethon is chatty at INFO; keep its internal noise out of our log.
    logging.getLogger("telethon").setLevel(logging.WARNING)


def load_environment() -> None:
    """Load ``.env`` (if present) without overriding real env variables."""
    if ENV_FILE.is_file():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
        LOGGER.debug("Loaded environment from %s", ENV_FILE)
    else:
        LOGGER.debug("No .env at %s; using process environment.", ENV_FILE)


def parse_chats(raw: str | None) -> list[str]:
    """Split a comma-separated chat list into cleaned, non-empty targets."""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_monitor_chats() -> list[str]:
    """Read the ``--live`` chat targets from the ``TG_MONITOR_CHATS`` env var."""
    return parse_chats((os.getenv(MONITOR_CHATS_ENV) or "").strip())


def read_session_name() -> str:
    """Return the Telethon session name from ``TG_SESSION_NAME``.

    Falls back to :data:`DEFAULT_SESSION` when the variable is unset or empty,
    matching :func:`auth_test.read_session_name` so both entry points reuse the
    same authorized session file.
    """
    return (os.getenv("TG_SESSION_NAME") or "").strip() or DEFAULT_SESSION


def preview(text: str, limit: int = 90) -> str:
    """Return a single-line, length-capped preview of ``text`` for logs."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #


def install_shutdown_handler(stop_event: asyncio.Event) -> bool:
    """Wire ``SIGINT``/``SIGTERM`` to set ``stop_event``.

    Returns ``True`` when at least one handler was installed. On Windows,
    ``loop.add_signal_handler`` raises ``NotImplementedError``; the fallback uses
    :func:`signal.signal`, which is safe because handlers run in the main thread.
    """
    loop = asyncio.get_running_loop()

    def _request_stop(*_args: Any) -> None:
        if stop_event.is_set():
            return
        LOGGER.info("Shutdown signal received — stopping gracefully...")
        loop.call_soon_threadsafe(stop_event.set)

    installed = False
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
            installed = True
            continue
        except (NotImplementedError, RuntimeError):
            pass
        try:
            signal.signal(sig, _request_stop)
            installed = True
        except (ValueError, OSError):  # pragma: no cover - platform specific
            LOGGER.debug("Could not install handler for %s", name)

    if installed:
        LOGGER.debug("Graceful shutdown handlers installed.")
    else:
        LOGGER.debug("No signal handlers installed; relying on KeyboardInterrupt.")
    return installed


# --------------------------------------------------------------------------- #
# Demo mode
# --------------------------------------------------------------------------- #


async def run_demo(db_path: str, delay: float) -> int:
    """Feed the sample messages through the pipeline with delays between them."""
    await db_module.init_db(db_path)
    LOGGER.info(
        "Demo mode | messages=%d | delay=%.1fs | db=%s",
        len(DEMO_MESSAGES),
        delay,
        db_path,
    )

    counters: Counter[str] = Counter()
    try:
        for index, (text, source, link) in enumerate(DEMO_MESSAGES, start=1):
            LOGGER.info(
                "Demo %d/%d | source=%s | text=%r",
                index,
                len(DEMO_MESSAGES),
                source,
                preview(text),
            )
            result = await pipeline.process_raw_message(
                text, source, link, db_path
            )
            counters[result["status"]] += 1
            LOGGER.info("Demo %d/%d result: %s", index, len(DEMO_MESSAGES), result)

            if index < len(DEMO_MESSAGES) and delay > 0:
                await asyncio.sleep(delay)
    finally:
        await classifier.aclose()

    LOGGER.info(
        "Demo complete | dispatched=%d filtered=%d duplicate=%d "
        "rejected_by_llm=%d llm_unavailable=%d error=%d",
        counters[pipeline.STATUS_DISPATCHED],
        counters[pipeline.STATUS_FILTERED],
        counters[pipeline.STATUS_DUPLICATE],
        counters[pipeline.STATUS_REJECTED_BY_LLM],
        counters[pipeline.STATUS_LLM_UNAVAILABLE],
        counters[pipeline.STATUS_ERROR],
    )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Live mode
# --------------------------------------------------------------------------- #


def _chat_name(event: Any) -> str:
    """Derive a human-readable chat name from a Telethon ``NewMessage`` event."""
    chat = getattr(event, "chat", None)
    if chat is not None:
        title = getattr(chat, "title", None)
        if title:
            return str(title)
        username = getattr(chat, "username", None)
        if username:
            return f"@{username}"
        first_name = getattr(chat, "first_name", None)
        if first_name:
            return str(first_name)
    chat_id = getattr(event, "chat_id", None)
    return str(chat_id) if chat_id is not None else "unknown"


def _message_link(event: Any) -> str | None:
    """Build a ``t.me`` permalink for a Telethon ``NewMessage`` event."""
    message_id = getattr(event, "id", None)
    if message_id is None:
        return None

    chat = getattr(event, "chat", None)
    username = getattr(chat, "username", None) if chat is not None else None
    if username:
        return f"https://t.me/{username}/{message_id}"

    chat_id = getattr(event, "chat_id", None)
    if chat_id is None:
        return None
    marker = str(chat_id)
    if marker.startswith("-100"):
        marker = marker[4:]
    return f"https://t.me/c/{marker}/{message_id}"


def _entity_label(entity: Any) -> str:
    """Return a human-readable label for a resolved Telethon input entity."""
    username = getattr(entity, "username", None)
    if username:
        return f"@{username}"
    title = getattr(entity, "title", None)
    if title:
        return str(title)
    for attr in ("user_id", "channel_id", "chat_id"):
        value = getattr(entity, attr, None)
        if value is not None:
            return f"{attr.replace('_id', '')}:{value}"
    return str(entity)


async def resolve_valid_chat_entities(
    client: Any, chat_identifiers: Sequence[str]
) -> list[Any]:
    """Resolve chat identifiers to Telethon input entities, skipping invalid ones.

    A single non-existent or inaccessible username must never prevent the other
    monitored chats from being dispatched, so every resolution failure is logged
    and skipped instead of propagating out of startup.
    """
    from telethon.errors import ChannelPrivateError, UsernameNotOccupiedError

    valid_entities: list[Any] = []
    skipped: list[str] = []

    for identifier in chat_identifiers:
        try:
            entity = await client.get_input_entity(identifier)
        except (UsernameNotOccupiedError, ChannelPrivateError, ValueError) as exc:
            LOGGER.warning(
                "Skipping chat %r: could not resolve (%s: %s)",
                identifier,
                type(exc).__name__,
                exc,
            )
            skipped.append(identifier)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad identifier must not abort startup
            LOGGER.warning(
                "Skipping chat %r: could not resolve (%s: %s)",
                identifier,
                type(exc).__name__,
                exc,
            )
            skipped.append(identifier)
            continue
        valid_entities.append(entity)
        LOGGER.info("Resolved chat %r -> %s", identifier, _entity_label(entity))

    if valid_entities:
        LOGGER.info(
            "Active resolved chats (%d/%d): %s",
            len(valid_entities),
            len(chat_identifiers),
            ", ".join(_entity_label(entity) for entity in valid_entities),
        )
    else:
        LOGGER.warning(
            "No chat entities resolved from %d identifier(s); "
            "listening to all incoming messages.",
            len(chat_identifiers),
        )
    if skipped:
        LOGGER.warning(
            "Skipped %d invalid chat(s): %s", len(skipped), ", ".join(skipped)
        )
    return valid_entities


async def run_live(db_path: str, session: str, chats: list[str]) -> int:
    """Start the Telethon client and process incoming messages until stopped."""
    # Imported lazily so ``--demo`` works even without Telethon installed.
    from auth_test import build_proxy, read_api_credentials
    from telethon import TelegramClient, events

    api_id, api_hash = read_api_credentials()
    proxy = build_proxy()
    client = TelegramClient(session, api_id, api_hash, proxy=proxy)

    await db_module.init_db(db_path)

    stop_event = asyncio.Event()

    try:
        LOGGER.info(
            "Starting Telethon listener | session=%s | chats=%s",
            session,
            ", ".join(chats) if chats else "<all incoming>",
        )
        await client.start()

        valid_entities = await resolve_valid_chat_entities(client, chats)
        chats_filter = valid_entities if valid_entities else None

        @client.on(events.NewMessage(chats=chats_filter))
        async def _on_new_message(event: Any) -> None:
            text = getattr(event, "raw_text", "") or ""
            if not text.strip():
                LOGGER.debug(
                    "Skipping empty/non-text message from %s", _chat_name(event)
                )
                return
            source = _chat_name(event)
            link = _message_link(event)
            try:
                result = await pipeline.process_raw_message(
                    text, source, link, db_path
                )
            except Exception:  # pragma: no cover - process_raw_message is defensive
                LOGGER.exception("Unhandled pipeline failure | source=%s", source)
                return
            LOGGER.info(
                "Live result | source=%s | status=%s", source, result["status"]
            )

        install_shutdown_handler(stop_event)
        LOGGER.info("Listening for new messages. Press Ctrl+C to stop.")

        disconnect_task = asyncio.create_task(client.run_until_disconnected())
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            await asyncio.wait(
                {disconnect_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (disconnect_task, stop_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                disconnect_task, stop_task, return_exceptions=True
            )
    finally:
        if client.is_connected():
            await client.disconnect()
            LOGGER.info("Disconnected from Telegram.")
        await classifier.aclose()

    LOGGER.info("Live mode stopped cleanly.")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Argument parsing / entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="lead-sniper",
        description="Lead Sniper — Telegram B2B lead pipeline.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run the offline sample-message simulation through the pipeline.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Start the Telethon listener for incoming messages.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Deduplication SQLite cache (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Telethon session name. Defaults to the TG_SESSION_NAME environment "
            f"variable, falling back to {DEFAULT_SESSION}."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between demo messages in seconds (default: {DEFAULT_DELAY}).",
    )
    parser.add_argument(
        "--chats",
        default=None,
        help=(
            "Comma-separated chat usernames/ids for --live. "
            f"Overrides the {MONITOR_CHATS_ENV} environment variable."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch the requested mode and translate failures."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    load_environment()

    try:
        if args.demo:
            return asyncio.run(run_demo(args.db, max(0.0, args.delay)))

        chats = parse_chats(args.chats) or read_monitor_chats()
        session = args.session or read_session_name()
        return asyncio.run(run_live(args.db, session, chats))

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user (Ctrl+C). Shutting down.")
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI entry point
        # ConfigError lives in auth_test and is only imported in live mode; a
        # missing/invalid .env surfaces here as a generic, logged failure.
        LOGGER.error("Fatal error: %s", exc)
        LOGGER.debug("Traceback:", exc_info=True)
        return EXIT_CONFIG_ERROR if type(exc).__name__ == "ConfigError" else EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
