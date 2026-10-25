"""Sprint 1.3 — isolated Telegram authentication smoke test.

Purpose
-------
Prove that the Telegram API credentials in ``.env`` are valid and that the
account can be authorized from this machine (optionally through a SOCKS5
proxy) *before* any scraping logic is written. The resulting session file is
reused by the rest of the pipeline.

Usage
-----
1. Fill in ``.env`` (see ``.env.example``).
2. Run ``python auth_test.py`` in an interactive terminal.
3. Enter the phone number, the login code Telegram sends you and, if enabled,
   your 2FA password.

On success the authorized session is persisted to
``session_auth_test.session`` and the process exits with code ``0``.

Exit codes
----------
0  Authentication succeeded.
1  Configuration problem (missing/invalid ``.env`` values).
2  Authentication failed or was aborted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from telethon import TelegramClient

try:  # PySocks is only required when a proxy is actually configured.
    import socks
except ImportError:  # pragma: no cover - exercised only on broken installs
    socks = None  # type: ignore[assignment]

LOGGER = logging.getLogger("lead_sniper.auth_test")

SESSION_NAME: str = "session_auth_test"
SESSION_FILE: str = f"{SESSION_NAME}.session"
ENV_FILE: Path = Path(__file__).resolve().with_name(".env")

# Telethon proxy tuple: (proxy_type, addr, port, rdns, username, password)
ProxyConfig = Optional[Tuple[int, str, int, bool, Optional[str], Optional[str]]]

EXIT_OK: int = 0
EXIT_CONFIG_ERROR: int = 1
EXIT_AUTH_ERROR: int = 2

# Hosts that mean "no proxy" — a local SOCKS listener is not a real egress.
_DIRECT_HOSTS: frozenset[str] = frozenset({"", "127.0.0.1", "localhost"})


class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


def configure_logging() -> None:
    """Configure a single, readable log format for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def load_environment() -> None:
    """Load ``.env`` without blocking the event loop.

    ``python-dotenv`` performs synchronous file I/O, so it is dispatched to a
    worker thread. Existing process environment variables win (``override``
    is ``False``), which keeps CI/container overrides working.
    """
    if not ENV_FILE.is_file():
        raise ConfigError(
            f"Environment file not found: {ENV_FILE}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    loaded = await asyncio.to_thread(
        load_dotenv, dotenv_path=ENV_FILE, override=False
    )
    if not loaded:
        LOGGER.warning("No variables were loaded from %s", ENV_FILE)


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


def build_proxy() -> ProxyConfig:
    """Build a Telethon SOCKS5 proxy tuple, or ``None`` for a direct connection.

    A proxy is used only when ``PROXY_HOST`` is populated with something other
    than a loopback address.

    Raises:
        ConfigError: if a proxy is requested but ``PySocks`` is unavailable or
            ``PROXY_PORT`` is not a valid port number.
    """
    host = _read("PROXY_HOST")
    if host.lower() in _DIRECT_HOSTS:
        LOGGER.info("No proxy configured — connecting directly to Telegram.")
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

    LOGGER.info(
        "Routing traffic through SOCKS5 proxy %s:%d (auth=%s).",
        host,
        port,
        "yes" if username else "no",
    )
    # rdns=True lets the proxy resolve DNS, avoiding local DNS leaks.
    return (socks.SOCKS5, host, port, True, username, password)


async def authenticate() -> int:
    """Run the interactive login flow and report the authorized account."""
    await load_environment()
    api_id, api_hash = read_api_credentials()
    proxy = build_proxy()

    client = TelegramClient(SESSION_NAME, api_id, api_hash, proxy=proxy)
    try:
        LOGGER.info(
            "Starting interactive Telegram login (session=%s)...", SESSION_FILE
        )
        # start() prompts for phone number, login code and 2FA password as
        # needed, and reuses the saved session when it is still valid.
        await client.start()

        me = await client.get_me()
        if me is None:
            LOGGER.error("Login completed but no user profile was returned.")
            return EXIT_AUTH_ERROR

        username = f"@{me.username}" if me.username else "(not set)"
        print()
        print("[SUCCESS] Authenticated successfully!")
        print(f"User ID: {me.id}")
        print(f"Username: {username}")
        print(f"Name: {me.first_name or ''}")
        print(f"Session saved: {SESSION_FILE}")
        return EXIT_OK
    finally:
        if client.is_connected():
            await client.disconnect()
            LOGGER.info("Disconnected from Telegram.")


def main() -> int:
    """Synchronous CLI wrapper translating failures into exit codes."""
    configure_logging()
    try:
        return asyncio.run(authenticate())
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return EXIT_CONFIG_ERROR
    except (EOFError, KeyboardInterrupt):
        LOGGER.error(
            "Login aborted: no interactive terminal available, or cancelled by user."
        )
        return EXIT_AUTH_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI entry point
        LOGGER.error("Authentication failed: %s", exc)
        LOGGER.debug("Traceback:", exc_info=True)
        return EXIT_AUTH_ERROR


if __name__ == "__main__":
    sys.exit(main())
