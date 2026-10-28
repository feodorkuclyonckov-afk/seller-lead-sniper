"""Sprint 1.3 — isolated Telegram authentication smoke test (QR login).

Purpose
-------
Prove that the Telegram API credentials in ``.env`` are valid and that the
account can be authorized from this machine (optionally through a SOCKS5
proxy) *before* any scraping logic is written. Authentication is performed by
linking a desktop device via a QR code — no SMS code is required. The
resulting session file is reused by the rest of the pipeline.

Usage
-----
1. Fill in ``.env`` (see ``.env.example``).
2. Run ``python auth_test.py`` in an interactive terminal.
3. Scan the rendered QR code with your phone:
   Telegram -> Settings -> Devices -> Link Desktop Device.
4. If 2FA is enabled, enter your password when prompted.

On success the authorized session is persisted to ``<TG_SESSION_NAME>.session``
and the process exits with code ``0``.

Exit codes
----------
0  Authentication succeeded.
1  Configuration problem (missing/invalid ``.env`` values).
2  Authentication failed or was aborted.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import qrcode
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import PasswordHashInvalidError, SessionPasswordNeededError

try:  # python-socks is only required when a proxy is actually configured.
    import python_socks  # noqa: F401
except ImportError:  # pragma: no cover - exercised only on broken installs
    python_socks = None  # type: ignore[assignment]

LOGGER = logging.getLogger("lead_sniper.auth_test")

DEFAULT_SESSION_NAME: str = "session_auth_test"
ENV_FILE: Path = Path(__file__).resolve().with_name(".env")

# Telethon proxy config: a python-socks style dict, or None for direct.
ProxyConfig = Optional[dict]

EXIT_OK: int = 0
EXIT_CONFIG_ERROR: int = 1
EXIT_AUTH_ERROR: int = 2

# Hosts that mean "no proxy" — a local SOCKS listener is not a real egress.
_DIRECT_HOSTS: frozenset[str] = frozenset({"", "127.0.0.1", "localhost"})

#: Proxy schemes understood by python-socks / Telethon.
_SUPPORTED_PROXY_TYPES: frozenset[str] = frozenset({"socks5", "socks4", "http"})

#: Default scheme. The configured provider only speaks HTTP CONNECT on the
#: port it advertises as SOCKS5, so ``http`` is the working default.
DEFAULT_PROXY_TYPE: str = "http"


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


def read_session_name() -> str:
    """Return the Telethon session name from ``TG_SESSION_NAME``.

    Falls back to :data:`DEFAULT_SESSION_NAME` when the variable is unset.
    """
    return _read("TG_SESSION_NAME") or DEFAULT_SESSION_NAME


def read_proxy_type() -> str:
    """Return the configured proxy scheme, defaulting to ``http``.

    The provider behind ``PROXY_HOST`` advertises port 8000 as SOCKS5 but only
    speaks HTTP ``CONNECT``: a SOCKS5 handshake is silently dropped, surfacing
    as ``ProxyTimeoutError: Proxy connection timed out``, while HTTP CONNECT
    succeeds in well under a second. ``http`` is therefore the safe default;
    set ``PROXY_TYPE=socks5`` explicitly for a genuine SOCKS5 endpoint.

    Raises:
        ConfigError: if ``PROXY_TYPE`` is set to an unsupported scheme.
    """
    raw = _read("PROXY_TYPE").lower() or DEFAULT_PROXY_TYPE
    if raw not in _SUPPORTED_PROXY_TYPES:
        raise ConfigError(
            "PROXY_TYPE must be one of "
            f"{sorted(_SUPPORTED_PROXY_TYPES)}, got {raw!r}."
        )
    return raw


def build_proxy() -> ProxyConfig:
    """Build a Telethon proxy dict, or ``None`` for a direct connection.

    A proxy is used only when ``PROXY_HOST`` is populated with something other
    than a loopback address.

    ``rdns`` is deliberately ``False``: Telethon then connects to Telegram's
    numeric DC IP addresses directly through the proxy. Remote DNS resolution
    (``rdns=True``) times out on this provider.

    Raises:
        ConfigError: if a proxy is requested but ``python-socks`` is unavailable
            or ``PROXY_PORT`` is not a valid port number.
    """
    host = _read("PROXY_HOST")
    if host.lower() in _DIRECT_HOSTS:
        LOGGER.info("No proxy configured — connecting directly to Telegram.")
        return None

    if python_socks is None:
        raise ConfigError(
            "PROXY_HOST is set but python-socks is not installed. "
            'Run: pip install "python-socks[asyncio]"'
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
    proxy_type = read_proxy_type()

    LOGGER.info(
        "Routing traffic through %s proxy %s:%d (auth=%s, rdns=off).",
        proxy_type.upper(),
        host,
        port,
        "yes" if username else "no",
    )
    proxy: dict = {
        "proxy_type": proxy_type,
        "addr": host,
        "port": port,
        "rdns": False,
    }
    if username:
        proxy["username"] = username
    if password:
        proxy["password"] = password
    return proxy


def render_qr(url: str) -> None:
    """Render a scannable QR code for ``url`` in the terminal."""
    qr = qrcode.QRCode()
    qr.add_data(url)
    qr.print_ascii(invert=True)


def print_qr_instructions() -> None:
    """Print human-readable instructions for linking a desktop device."""
    print()
    print("=" * 60)
    print("  Scan the QR code above with your phone:")
    print("  Telegram -> Settings -> Devices -> Link Desktop Device")
    print("=" * 60)
    print()


async def qr_authenticate(client: TelegramClient) -> None:
    """Run the QR login flow until the client is authorized.

    Raises:
        SessionPasswordNeededError: re-raised only if the 2FA password is
            invalid (handled by the caller as an auth failure).
    """
    while True:
        qr_login = await client.qr_login()
        render_qr(qr_login.url)
        print_qr_instructions()

        try:
            await qr_login.wait()
            return
        except SessionPasswordNeededError:
            password = getpass.getpass("Enter 2FA password: ")
            try:
                await client.sign_in(password=password)
            except PasswordHashInvalidError:
                LOGGER.error("Invalid 2FA password. Authentication aborted.")
                raise
            return
        except asyncio.TimeoutError:
            # The QR code expired before it was scanned — issue a fresh one.
            LOGGER.warning("QR code expired. Generating a new one...")
            continue


async def authenticate() -> int:
    """Run the QR login flow and report the authorized account."""
    await load_environment()
    api_id, api_hash = read_api_credentials()
    session_name = read_session_name()
    session_file = f"{session_name}.session"
    proxy = build_proxy()

    client = TelegramClient(session_name, api_id, api_hash, proxy=proxy)
    try:
        LOGGER.info("Connecting to Telegram (session=%s)...", session_file)
        await client.connect()

        if not await client.is_user_authorized():
            LOGGER.info("Not authorized — starting QR login flow.")
            await qr_authenticate(client)

        me = await client.get_me()
        if me is None:
            LOGGER.error("Login completed but no user profile was returned.")
            return EXIT_AUTH_ERROR

        full_name = " ".join(
            part for part in (me.first_name, me.last_name) if part
        ) or "(not set)"
        username = f"@{me.username}" if me.username else "(not set)"

        print()
        print("[SUCCESS] Authenticated successfully!")
        print(f"Name: {full_name}")
        print(f"Username: {username}")
        print(f"User ID: {me.id}")
        print(f"Session saved: {session_file}")
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
    except PasswordHashInvalidError:
        LOGGER.error("Authentication failed: invalid 2FA password.")
        return EXIT_AUTH_ERROR
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
