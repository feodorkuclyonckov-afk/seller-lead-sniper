"""Safely join curated public B2B/e-commerce Telegram chats.

Purpose
-------
Bootstrap the ``TG_MONITOR_CHATS`` list used by ``python main.py --live`` by
joining a small, curated set of public B2B chats (cargo, e-commerce,
fulfillment, marketplaces, marketing). The script is deliberately conservative:

* it reuses the authorized Telethon session created by ``auth_test.py``;
* it never crashes on a single bad/private/renamed chat;
* it honours ``FloodWaitError`` and backs off;
* it sleeps 15-25 seconds between successful joins to avoid spam triggers;
* it rewrites only the ``TG_MONITOR_CHATS`` line in ``.env``.

Usage
-----
1. Run ``python auth_test.py`` once to authorize the session.
2. Run ``python join_chats.py``.
3. Start the listener with ``python main.py --live``.

Exit codes
----------
0  At least one chat was joined/verified and ``.env`` was updated.
1  Configuration problem (missing/invalid ``.env`` values).
2  The Telegram session is not authorized.
3  No chats were joined/verified; ``.env`` was left unchanged.
4  Unexpected runtime failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    UserAlreadyParticipantError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.tl.functions.channels import JoinChannelRequest

try:  # PySocks is only required when a proxy is actually configured.
    import socks
except ImportError:  # pragma: no cover - exercised only on broken installs
    socks = None  # type: ignore[assignment]

LOGGER = logging.getLogger("lead_sniper.join_chats")

ENV_FILE: Path = Path(__file__).resolve().with_name(".env")
DEFAULT_SESSION_NAME: str = "session_auth_test"
MONITOR_CHATS_ENV: str = "TG_MONITOR_CHATS"

#: Curated public B2B/e-commerce chats. Usernames are stored without ``@``.
#: Edit this tuple to target a different niche.
TARGET_CHATS: Tuple[str, ...] = (
    "wb_chat_sellers",
    "cargo_china_chat",
    "fulfillment_russia",
    "marketplace_chat_wb_ozon",
    "cargo_chat_moscow",
    "ecom_russia",
)

#: Delay range (seconds) between successful joins.
MIN_JOIN_DELAY: float = 15.0
MAX_JOIN_DELAY: float = 25.0

#: A flood wait longer than this is treated as "skip this chat".
MAX_FLOOD_WAIT: int = 300

#: How many times a single chat is attempted after a tolerable flood wait.
MAX_JOIN_ATTEMPTS: int = 2

OUTCOME_JOINED: str = "joined"
OUTCOME_ALREADY: str = "already"
OUTCOME_SKIPPED: str = "skipped"

EXIT_OK: int = 0
EXIT_CONFIG_ERROR: int = 1
EXIT_AUTH_ERROR: int = 2
EXIT_NO_CHATS: int = 3
EXIT_RUNTIME_ERROR: int = 4

# Hosts that mean "no proxy" — a local SOCKS listener is not a real egress.
_DIRECT_HOSTS: frozenset[str] = frozenset({"", "127.0.0.1", "localhost"})

# Telethon proxy tuple: (proxy_type, addr, port, rdns, username, password)
ProxyConfig = Optional[Tuple[int, str, int, bool, Optional[str], Optional[str]]]


class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


class AuthError(RuntimeError):
    """Raised when the Telethon session is not authorized."""


def configure_logging() -> None:
    """Configure a single, readable log format for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Telethon is chatty at INFO; keep its internal noise out of our log.
    logging.getLogger("telethon").setLevel(logging.WARNING)


def load_environment() -> None:
    """Load ``.env`` without overriding real process environment variables."""
    if not ENV_FILE.is_file():
        raise ConfigError(
            f"Environment file not found: {ENV_FILE}. "
            "Copy .env.example to .env and fill in your credentials."
        )
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _read(name: str) -> str:
    """Return a stripped environment value, or an empty string if unset."""
    return (os.getenv(name) or "").strip()


def read_api_credentials() -> Tuple[int, str]:
    """Validate and return ``(api_id, api_hash)``.

    Raises:
        ConfigError: if either value is missing or ``TG_API_ID`` is not a
            positive integer.
    """
    raw_api_id = _read("TG_API_ID")
    api_hash = _read("TG_API_HASH")

    missing = [
        name
        for name, value in (("TG_API_ID", raw_api_id), ("TG_API_HASH", api_hash))
        if not value
    ]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + f". Populate them in {ENV_FILE} (see .env.example)."
        )

    try:
        api_id = int(raw_api_id)
    except ValueError as exc:
        raise ConfigError(
            f"TG_API_ID must be an integer, got {raw_api_id!r}."
        ) from exc

    if api_id <= 0:
        raise ConfigError(f"TG_API_ID must be a positive integer, got {api_id}.")

    return api_id, api_hash


def read_session_name() -> str:
    """Return the Telethon session name from ``TG_SESSION_NAME``."""
    return _read("TG_SESSION_NAME") or DEFAULT_SESSION_NAME


def build_proxy() -> ProxyConfig:
    """Build a Telethon SOCKS5 proxy tuple, or ``None`` for a direct connection."""
    host = _read("PROXY_HOST")
    if host.lower() in _DIRECT_HOSTS:
        return None

    if socks is None:
        raise ConfigError(
            "PROXY_HOST is set but PySocks is not installed. "
            "Run: pip install -r requirements.txt"
        )

    raw_port = _read("PROXY_PORT")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ConfigError(
            f"PROXY_PORT must be an integer, got {raw_port!r}."
        ) from exc

    if not 0 < port < 65536:
        raise ConfigError(f"PROXY_PORT out of range: {port}.")

    username = _read("PROXY_USER") or None
    password = _read("PROXY_PASS") or None

    LOGGER.info("Routing traffic through SOCKS5 proxy %s:%d.", host, port)
    # rdns=True lets the proxy resolve DNS, avoiding local DNS leaks.
    return (socks.SOCKS5, host, port, True, username, password)


async def _join_one(client: TelegramClient, username: str) -> str:
    """Attempt to join ``username`` and return an outcome constant.

    ``client.get_entity`` resolves the public username first; the actual join
    is performed with ``JoinChannelRequest``. Existing membership surfaces as
    ``UserAlreadyParticipantError``, which is the canonical "already joined"
    signal in Telethon.
    """
    for attempt in range(1, MAX_JOIN_ATTEMPTS + 1):
        try:
            entity = await client.get_entity(username)
            await client(JoinChannelRequest(entity))
            return OUTCOME_JOINED
        except UserAlreadyParticipantError:
            return OUTCOME_ALREADY
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            if seconds > MAX_FLOOD_WAIT:
                LOGGER.warning(
                    "Flood wait of %ss for %s exceeds the %ss limit; skipping.",
                    seconds,
                    username,
                    MAX_FLOOD_WAIT,
                )
                return OUTCOME_SKIPPED
            if attempt >= MAX_JOIN_ATTEMPTS:
                LOGGER.warning(
                    "Flood wait persists for %s; skipping after %d attempt(s).",
                    username,
                    attempt,
                )
                return OUTCOME_SKIPPED
            LOGGER.warning(
                "Flood wait: sleeping %ss before retrying %s (attempt %d/%d).",
                seconds,
                username,
                attempt,
                MAX_JOIN_ATTEMPTS,
            )
            await asyncio.sleep(seconds)
        except (
            ChannelPrivateError,
            UsernameInvalidError,
            UsernameNotOccupiedError,
        ) as exc:
            LOGGER.warning("Cannot join %s: %s", username, exc)
            return OUTCOME_SKIPPED
        except Exception as exc:  # noqa: BLE001 - one bad chat must not abort the run
            LOGGER.warning("Skipping %s: %s", username, exc)
            return OUTCOME_SKIPPED

    return OUTCOME_SKIPPED


async def join_targets(client: TelegramClient) -> Tuple[list[str], list[str]]:
    """Join every target chat, returning ``(joined_or_verified, skipped)``."""
    joined: list[str] = []
    skipped: list[str] = []
    total = len(TARGET_CHATS)

    for index, username in enumerate(TARGET_CHATS, start=1):
        LOGGER.info("Processing %d/%d: %s", index, total, username)
        outcome = await _join_one(client, username)

        if outcome == OUTCOME_JOINED:
            print(f"[+] Successfully joined: {username}")
            joined.append(username)
        elif outcome == OUTCOME_ALREADY:
            print(f"[*] Already member: {username}")
            joined.append(username)
        else:
            print(f"[-] Skipped: {username}")
            skipped.append(username)

        # Only actual joins need a cooldown; already-member checks are cheap.
        if outcome == OUTCOME_JOINED and index < total:
            delay = random.uniform(MIN_JOIN_DELAY, MAX_JOIN_DELAY)
            LOGGER.info("Sleeping %.1fs before the next join...", delay)
            await asyncio.sleep(delay)

    return joined, skipped


def update_env_file(env_path: Path, usernames: Sequence[str]) -> str:
    """Replace or append ``TG_MONITOR_CHATS`` in ``env_path``.

    Existing variables, comments and line endings are preserved; only the
    ``TG_MONITOR_CHATS`` line is touched. Returns the new value.
    """
    value = ",".join(usernames)
    new_line = f"{MONITOR_CHATS_ENV}={value}"
    pattern = re.compile(
        rf"^\s*(?:export\s+)?{re.escape(MONITOR_CHATS_ENV)}\s*="
    )

    text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)

    for index, line in enumerate(lines):
        if pattern.match(line):
            ending = newline if line.endswith(("\n", "\r")) else ""
            lines[index] = new_line + ending
            break
    else:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline
        lines.append(new_line + newline)

    new_text = "".join(lines)
    if not new_text.endswith(("\n", "\r")):
        new_text += newline

    with env_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(new_text)

    # Keep the in-process environment consistent with the file on disk.
    os.environ[MONITOR_CHATS_ENV] = value
    return value


async def run() -> int:
    """Connect, join the target chats and persist the result to ``.env``."""
    load_environment()
    api_id, api_hash = read_api_credentials()
    session_name = read_session_name()
    proxy = build_proxy()

    client = TelegramClient(session_name, api_id, api_hash, proxy=proxy)
    try:
        LOGGER.info("Connecting to Telegram (session=%s.session)...", session_name)
        await client.connect()

        if not await client.is_user_authorized():
            raise AuthError(
                "Telegram session is not authorized. Run `python auth_test.py` "
                "first to link your account, then re-run this script."
            )

        me = await client.get_me()
        if me is not None:
            display = " ".join(
                part for part in (me.first_name, me.last_name) if part
            )
            LOGGER.info(
                "Authorized as %s (@%s, id=%s).",
                display or "(no name)",
                me.username or "no_username",
                me.id,
            )

        joined, skipped = await join_targets(client)
    finally:
        if client.is_connected():
            await client.disconnect()
            LOGGER.info("Disconnected from Telegram.")

    if not joined:
        print()
        print("[-] No chats were joined or verified; .env was left unchanged.")
        return EXIT_NO_CHATS

    value = update_env_file(ENV_FILE, joined)

    print()
    print(f"[*] Updated {MONITOR_CHATS_ENV}={value}")
    print(f"[*] .env updated: {ENV_FILE}")
    if skipped:
        print(f"[*] Skipped {len(skipped)} chat(s): {', '.join(skipped)}")
    return EXIT_OK


def main() -> int:
    """Synchronous CLI wrapper translating failures into exit codes."""
    configure_logging()
    try:
        return asyncio.run(run())
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return EXIT_CONFIG_ERROR
    except AuthError as exc:
        LOGGER.error("%s", exc)
        return EXIT_AUTH_ERROR
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return EXIT_RUNTIME_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI entry point
        LOGGER.error("Unexpected failure: %s", exc)
        LOGGER.debug("Traceback:", exc_info=True)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
