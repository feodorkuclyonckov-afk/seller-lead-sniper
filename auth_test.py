"""Sprint 1.3b — isolated Telegram authentication smoke test (phone login).

Purpose
-------
Prove that the Telegram API credentials in ``.env`` are valid and that the
account can be authorized from this machine (optionally through a SOCKS5 or
HTTP proxy) *before* any scraping logic is written. Authentication is
performed with an international phone number: Telegram sends a login code to
the Telegram app (or by SMS) and, if two-step verification is enabled, the
cloud password is requested. The resulting session file is reused by the rest
of the pipeline.

Usage
-----
1. Fill in ``.env`` (see ``.env.example``).
2. Run ``python auth_test.py`` in an interactive terminal and type the phone
   number in international format (``+7...``). The number may also be supplied
   with ``--phone`` or via ``TG_PHONE`` in ``.env``.
3. Enter the login code Telegram sends to your device.
4. If 2FA is enabled, enter your cloud password (input is hidden).

QR login (no login code, no phone number)
-----------------------------------------
``python auth_test.py --qr`` links the account through a QR token instead of a
phone code. The token URL is rendered with ``qrcode`` to ``qr.png`` in the
project root, so it can be scanned from any image viewer (terminal ASCII
rendering is deliberately not used). Scan it with
Telegram -> Settings -> Devices -> Link Desktop Device (or the "Scan QR code"
button on the device list). The token expires after ~30 seconds; the script
recreates it in a loop and overwrites ``qr.png`` each time. If two-step
verification is enabled, the cloud password is prompted (or taken from
``--password``). On success ``qr.png`` is deleted and the session file is
saved.

Non-interactive shells (CI, agent terminals without stdin) use a two-step
handshake instead, because ``phone_code_hash`` is bound to the auth key in the
session file rather than to the process:

1. ``python auth_test.py --request-code`` — sends the code, saves
   ``.auth_pending.json`` and exits.
2. ``python auth_test.py --code 12345`` (add ``--password ...`` if 2FA is on) —
   submits the code against the pending request and verifies the session.

Use ``--fresh`` to discard a stale/corrupt session file before logging in; the
previous file is renamed to ``<TG_SESSION_NAME>.session.stale.bak``.

On success the authorized session is persisted to ``<TG_SESSION_NAME>.session``,
verified to be non-empty, and the process exits with code ``0``.

Exit codes
----------
0  Authentication succeeded and the session file was verified.
1  Configuration problem (missing/invalid ``.env`` values or phone number).
2  Authentication failed or was aborted.
3  Authentication reported success but the session file is missing/empty.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberBannedError,
    PhoneNumberInvalidError,
    PhoneNumberUnoccupiedError,
    SessionPasswordNeededError,
)
from telethon.tl.types.auth import SentCode

try:  # python-socks is only required when a proxy is actually configured.
    import python_socks  # noqa: F401
except ImportError:  # pragma: no cover - exercised only on broken installs
    python_socks = None  # type: ignore[assignment]

try:  # qrcode is only required for the --qr login flow.
    import qrcode
except ImportError:  # pragma: no cover - exercised only on broken installs
    qrcode = None  # type: ignore[assignment]

LOGGER = logging.getLogger("lead_sniper.auth_test")

DEFAULT_SESSION_NAME: str = "session_auth_test"
ENV_FILE: Path = Path(__file__).resolve().with_name(".env")

#: Pending login state (phone + ``phone_code_hash``) written by one run and
#: consumed by the next, so the code can be submitted non-interactively.
PENDING_STATE_FILE: Path = Path(__file__).resolve().with_name(".auth_pending.json")

#: PNG the ``--qr`` login token is rendered to. It lives in the project root so
#: the user can open it in any image viewer instead of scanning the terminal.
QR_IMAGE_FILE: Path = Path(__file__).resolve().with_name("qr.png")

#: Render settings: module size in pixels and the mandatory white quiet zone.
QR_IMAGE_BOX_SIZE: int = 10
QR_IMAGE_BORDER: int = 4

#: How long one QR token is awaited before it is recreated. Telegram tokens
#: live for about 30 seconds; waiting 25 leaves room for the refresh round-trip.
QR_WAIT_TIMEOUT: float = 25.0

# Telethon proxy config: a python-socks style dict, or None for direct.
ProxyConfig = Optional[dict]

EXIT_OK: int = 0
EXIT_CONFIG_ERROR: int = 1
EXIT_AUTH_ERROR: int = 2
EXIT_SESSION_ERROR: int = 3

#: Interactive prompt texts (kept as constants so tests can assert on them).
PHONE_PROMPT: str = "Enter your phone number in international format (+7...): "
CODE_PROMPT: str = "Enter the login code sent by Telegram: "
CODE_RETRY_PROMPT: str = "Code rejected. Re-enter the login code: "
PASSWORD_PROMPT: str = "Two-step verification password: "
PASSWORD_RETRY_PROMPT: str = "Password rejected. Try again: "

#: How many times a bad login code / cloud password is retried.
MAX_CODE_ATTEMPTS: int = 3
MAX_PASSWORD_ATTEMPTS: int = 3

#: E.164: ``+``, a non-zero country code and 7-14 more digits (8-15 total).
_PHONE_PATTERN: re.Pattern[str] = re.compile(r"^\+[1-9]\d{7,14}$")

#: Formatting characters that are silently stripped from a typed number.
_PHONE_SEPARATORS: re.Pattern[str] = re.compile(r"[\s\-().\u00a0]")

# Hosts that mean "no proxy" — a local SOCKS listener is not a real egress.
_DIRECT_HOSTS: frozenset[str] = frozenset({"", "127.0.0.1", "localhost"})

#: Proxy schemes understood by python-socks / Telethon.
_SUPPORTED_PROXY_TYPES: frozenset[str] = frozenset({"socks5", "socks4", "http"})

#: Default scheme. The configured provider only speaks HTTP CONNECT on the
#: port it advertises as SOCKS5, so ``http`` is the working default.
DEFAULT_PROXY_TYPE: str = "http"


class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


class AuthError(RuntimeError):
    """Raised when Telegram rejects the phone, code or cloud password."""


class SessionError(RuntimeError):
    """Raised when the authorized session file was not written to disk."""


class _CodeRejected(AuthError):
    """Internal: Telegram rejected the submitted login code."""


class _PasswordRejected(AuthError):
    """Internal: Telegram rejected the submitted cloud password."""


def configure_logging() -> None:
    """Configure a single, readable log format for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Telethon is chatty at INFO; keep its internal noise out of our log.
    logging.getLogger("telethon").setLevel(logging.WARNING)


async def load_environment() -> None:
    """Load ``.env`` without blocking the event loop.

    ``python-dotenv`` performs synchronous file I/O, so it is dispatched to a
    worker thread. ``override`` is ``True`` so the project ``.env`` is the
    single source of truth: a stale ``PROXY_HOST`` exported by the shell (or
    inherited from a previous run) must not silently re-enable a proxy that
    the file explicitly leaves empty.
    """
    if not ENV_FILE.is_file():
        raise ConfigError(
            f"Environment file not found: {ENV_FILE}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    loaded = await asyncio.to_thread(
        load_dotenv, dotenv_path=ENV_FILE, override=True
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


def normalize_phone(raw: str) -> str:
    """Normalize a human-typed phone number to E.164 form.

    Spaces, parentheses, dots and dashes are stripped; the leading ``+`` is
    mandatory because Telegram needs the country code.

    Raises:
        ConfigError: if the number is empty, lacks ``+`` or does not match
            E.164 (8-15 digits, country code not starting with zero).
    """
    candidate = _PHONE_SEPARATORS.sub("", (raw or "").strip())

    if not candidate or candidate == "+":
        raise ConfigError(
            "Phone number is empty. Expected international format, "
            "e.g. +79991234567."
        )
    if not candidate.startswith("+"):
        raise ConfigError(
            "Phone number must start with '+' (international format), "
            f"got {raw!r}."
        )
    if not _PHONE_PATTERN.match(candidate):
        raise ConfigError(
            f"Phone number {raw!r} is not a valid E.164 number "
            "(expected +<country><subscriber>, 8-15 digits in total)."
        )
    return candidate


def read_phone() -> str:
    """Return the normalized ``TG_PHONE`` value, or ``""`` when unset.

    Raises:
        ConfigError: if ``TG_PHONE`` is set but malformed.
    """
    raw = _read("TG_PHONE")
    return normalize_phone(raw) if raw else ""


def mask_phone(phone: str) -> str:
    """Mask all but the country prefix and last two digits for logging."""
    if len(phone) <= 5:
        return "***"
    return f"{phone[:3]}***{phone[-2:]}"


def _read_line(prompt: str) -> str:
    """Blocking prompt reader; runs in a worker thread.

    The prompt is written with an explicit ``flush`` because stdout is
    block-buffered when the process is not attached to a TTY, which would
    otherwise leave the user staring at a blank terminal while the script
    waits for input.

    Raises:
        EOFError: if stdin is closed (non-interactive run).
    """
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError(f"stdin closed while waiting for input at {prompt!r}.")
    return line.strip()


async def resolve_phone(cli_phone: Optional[str]) -> str:
    """Determine the phone number to log in with.

    Precedence: ``--phone`` argument, then ``TG_PHONE`` from the environment,
    then an interactive prompt.

    Raises:
        ConfigError: if no number is available or the supplied one is invalid.
    """
    if cli_phone:
        return normalize_phone(cli_phone)

    configured = read_phone()
    if configured:
        return configured

    return normalize_phone(await asyncio.to_thread(_read_line, PHONE_PROMPT))


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

    A proxy is used only when ``PROXY_HOST`` is a non-empty string (after
    stripping whitespace) that is not a loopback address. There is no
    hardcoded fallback host: an unset or empty ``PROXY_HOST`` always means a
    direct connection.

    ``rdns`` is deliberately ``False``: Telethon then connects to Telegram's
    numeric DC IP addresses directly through the proxy. Remote DNS resolution
    (``rdns=True``) times out on this provider.

    Raises:
        ConfigError: if a proxy is requested but ``python-socks`` is unavailable
            or ``PROXY_PORT`` is not a valid port number.
    """
    host = _read("PROXY_HOST")
    if not host or host.lower() in _DIRECT_HOSTS:
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


def session_file_path(session_name: str) -> Path:
    """Return the SQLite session path used by Telethon for ``session_name``."""
    name = (session_name or "").strip() or DEFAULT_SESSION_NAME
    return Path(name if name.endswith(".session") else f"{name}.session")


def verify_session_file(session_name: str) -> Path:
    """Return the session path after checking that it exists and is non-empty.

    Raises:
        SessionError: if the file is missing or zero bytes, which means the
            authorization was not persisted (and will not survive a restart).
    """
    path = session_file_path(session_name)
    if not path.is_file():
        raise SessionError(f"Session file was not written: {path}")
    if path.stat().st_size == 0:
        raise SessionError(f"Session file is empty (0 bytes): {path}")
    return path


def reset_session_files(session_name: str) -> None:
    """Back up and remove a stale session (plus SQLite journals).

    The previous file is renamed to ``<path>.stale.bak`` so a failed re-login
    never destroys the last known-good session.
    """
    path = session_file_path(session_name)
    if path.is_file():
        backup = path.with_name(f"{path.name}.stale.bak")
        path.replace(backup)
        LOGGER.info("Moved stale session %s -> %s", path.name, backup.name)
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = path.with_name(f"{path.name}{suffix}")
        if sidecar.is_file():
            sidecar.unlink()
            LOGGER.info("Removed stale session sidecar %s", sidecar.name)


def load_pending_state() -> dict:
    """Return the pending login state, or ``{}`` when there is none.

    The state records the phone number and ``phone_code_hash`` of the last
    code request so the code can be submitted by a later process: the hash is
    bound to the auth key stored in the session file, not to the process that
    requested it. A corrupt or unreadable file is treated as "no pending
    login".
    """
    try:
        raw = PENDING_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        LOGGER.warning(
            "Cannot read pending login state %s: %s", PENDING_STATE_FILE, exc
        )
        return {}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOGGER.warning(
            "Ignoring corrupt pending login state %s: %s", PENDING_STATE_FILE, exc
        )
        return {}
    if not isinstance(state, dict):
        LOGGER.warning(
            "Ignoring unexpected pending login state in %s.", PENDING_STATE_FILE
        )
        return {}
    return state


def save_pending_state(
    phone: str, phone_code_hash: str, needs_password: bool = False
) -> None:
    """Persist the pending login so a later run can submit the code."""
    state = {
        "phone": phone,
        "phone_code_hash": phone_code_hash,
        "needs_password": needs_password,
        "created_at": int(time.time()),
    }
    PENDING_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        os.chmod(PENDING_STATE_FILE, 0o600)
    except OSError as exc:  # pragma: no cover - Windows ACLs may refuse chmod
        LOGGER.debug("Could not restrict permissions on %s: %s", PENDING_STATE_FILE, exc)


def clear_pending_state() -> None:
    """Delete the pending login state if present."""
    try:
        PENDING_STATE_FILE.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        LOGGER.warning("Could not remove %s: %s", PENDING_STATE_FILE, exc)


async def request_login_code(client: TelegramClient, phone: str) -> SentCode:
    """Ask Telegram to send a login code for ``phone``.

    Raises:
        ConfigError: if Telegram rejects the phone number itself.
        AuthError: if the number is banned/unoccupied or Telegram floods us.
    """
    try:
        sent = await client.send_code_request(phone)
    except PhoneNumberInvalidError as exc:
        raise ConfigError(
            f"Telegram rejected the phone number {mask_phone(phone)}. "
            "Use international format, e.g. +79991234567."
        ) from exc
    except (PhoneNumberBannedError, PhoneNumberUnoccupiedError) as exc:
        raise AuthError(
            f"Phone number {mask_phone(phone)} cannot log in: {exc}"
        ) from exc
    except FloodWaitError as exc:
        raise AuthError(
            "Telegram rate limit hit while requesting the code; "
            f"retry in {int(getattr(exc, 'seconds', 0) or 0)}s."
        ) from exc

    LOGGER.info(
        "Login code requested for %s (delivery=%s).",
        mask_phone(phone),
        type(sent.type).__name__,
    )
    return sent


def prompt_login_code(attempt: int) -> str:
    """Read the login code from the terminal (blocking)."""
    return _read_line(CODE_PROMPT if attempt == 1 else CODE_RETRY_PROMPT)


def prompt_password(attempt: int) -> str:
    """Read the cloud password without echoing it (blocking)."""
    return getpass.getpass(PASSWORD_PROMPT if attempt == 1 else PASSWORD_RETRY_PROMPT)


async def _sign_in_code_once(
    client: TelegramClient, phone: str, code: str, phone_code_hash: str
) -> bool:
    """Submit one login code; return ``True`` when 2FA is required.

    Raises:
        _CodeRejected: if Telegram reports the code as invalid.
        AuthError: if the code expired (retrying the same code is pointless).
    """
    try:
        await client.sign_in(
            phone=phone, code=code, phone_code_hash=phone_code_hash
        )
        return False
    except SessionPasswordNeededError:
        return True
    except PhoneCodeInvalidError as exc:
        raise _CodeRejected("Telegram rejected the login code.") from exc
    except PhoneCodeExpiredError as exc:
        raise AuthError(
            "The login code expired before it was submitted. "
            "Re-run auth_test.py without --code to request a new one."
        ) from exc


async def submit_login_code(
    client: TelegramClient, phone: str, code: str, phone_code_hash: str
) -> bool:
    """Submit a single known login code (the non-interactive ``--code`` path).

    Raises:
        AuthError: if the code is rejected or expired.
    """
    try:
        return await _sign_in_code_once(client, phone, code, phone_code_hash)
    except _CodeRejected as exc:
        raise AuthError(f"{exc} Check the digits and try again.") from exc


async def sign_in_with_code(
    client: TelegramClient,
    phone: str,
    sent: SentCode,
    reader=prompt_login_code,
) -> bool:
    """Prompt for the login code until it is accepted.

    Args:
        client: Connected (but unauthorized) Telethon client.
        phone: E.164 phone number used to request the code.
        sent: Value returned by :func:`request_login_code`.
        reader: Callable taking the attempt number and returning a code; it is
            invoked in a worker thread so the event loop keeps running.

    Returns:
        ``True`` when Telegram requires the two-step verification password.

    Raises:
        AuthError: if the code is rejected after
            :data:`MAX_CODE_ATTEMPTS` tries or has expired.
    """
    for attempt in range(1, MAX_CODE_ATTEMPTS + 1):
        code = await asyncio.to_thread(reader, attempt)
        try:
            return await _sign_in_code_once(
                client, phone, code, sent.phone_code_hash
            )
        except _CodeRejected:
            LOGGER.warning(
                "Login code rejected (attempt %d/%d).", attempt, MAX_CODE_ATTEMPTS
            )

    raise AuthError(f"Login code rejected after {MAX_CODE_ATTEMPTS} attempts.")


async def _sign_in_password_once(client: TelegramClient, password: str) -> None:
    """Submit one cloud password.

    Raises:
        _PasswordRejected: if Telegram rejects the password.
    """
    try:
        await client.sign_in(password=password)
    except PasswordHashInvalidError as exc:
        raise _PasswordRejected("Invalid cloud password.") from exc


async def submit_password(client: TelegramClient, password: str) -> None:
    """Submit a known cloud password (the non-interactive ``--password`` path).

    Raises:
        AuthError: if the password is rejected.
    """
    try:
        await _sign_in_password_once(client, password)
    except _PasswordRejected as exc:
        raise AuthError(
            f"{exc} Check your two-step verification password."
        ) from exc


async def sign_in_with_password(
    client: TelegramClient, reader=prompt_password
) -> None:
    """Complete the two-step verification challenge interactively.

    Raises:
        AuthError: if the password is rejected after
            :data:`MAX_PASSWORD_ATTEMPTS` tries.
    """
    for attempt in range(1, MAX_PASSWORD_ATTEMPTS + 1):
        password = await asyncio.to_thread(reader, attempt)
        try:
            await _sign_in_password_once(client, password)
            return
        except _PasswordRejected:
            LOGGER.warning(
                "Invalid 2FA password (attempt %d/%d).",
                attempt,
                MAX_PASSWORD_ATTEMPTS,
            )

    raise AuthError(
        f"Two-step verification password rejected after "
        f"{MAX_PASSWORD_ATTEMPTS} attempts."
    )


async def phone_authenticate(
    client: TelegramClient,
    phone: str,
    code: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Run the phone login flow until the client is authorized.

    Two modes are supported:

    * **Interactive** (``code`` is ``None``): a code is requested and read from
      the terminal; the cloud password is prompted with hidden input.
    * **Two-step / non-interactive** (``--code`` given): the code is submitted
      against the pending state saved by the previous run — no new code is
      sent — and ``--password`` completes the 2FA challenge.

    Args:
        client: Connected (but unauthorized) Telethon client.
        phone: E.164 phone number to log in with.
        code: Login code typed by the user, if known in advance.
        password: Cloud password, if two-step verification is enabled.

    Raises:
        ConfigError: if the phone number or the pending state is unusable.
        AuthError: if the code/password challenge cannot be completed.
    """
    if code is None and not sys.stdin.isatty():
        raise ConfigError(
            "No interactive terminal detected, so the login code cannot be "
            "prompted for. Run auth_test.py in a terminal, or pass --code "
            "(the code requested by the previous run)."
        )

    state = load_pending_state()
    phone_code_hash: Optional[str] = None

    if code:
        phone_code_hash = state.get("phone_code_hash")
        if not phone_code_hash or state.get("phone") != phone:
            raise ConfigError(
                "No pending login matches this phone number. Run "
                "`python auth_test.py` without --code to request a fresh code."
            )
        LOGGER.info(
            "Submitting the login code from the pending request for %s.",
            mask_phone(phone),
        )
        needs_password = await submit_login_code(
            client, phone, code, phone_code_hash
        )
    else:
        sent = await request_login_code(client, phone)
        phone_code_hash = sent.phone_code_hash
        save_pending_state(phone, phone_code_hash)
        needs_password = await sign_in_with_code(client, phone, sent)

    if needs_password:
        LOGGER.info("Two-step verification is enabled.")
        if password:
            await submit_password(client, password)
        elif sys.stdin.isatty():
            await sign_in_with_password(client)
        else:
            if phone_code_hash:
                save_pending_state(phone, phone_code_hash, needs_password=True)
            raise AuthError(
                "Two-step verification is enabled but no --password was given. "
                "The login code was accepted; re-run with --password to finish "
                "without requesting a new code."
            )

    if not await client.is_user_authorized():
        raise AuthError(
            "Telegram still reports the client as unauthorized after sign-in."
        )


def render_qr(url: str, path: Optional[Path] = None) -> Path:
    """Render ``url`` as a PNG QR code and return the file written.

    The image is written atomically (via a sibling ``.tmp`` file then
    ``replace``), so a viewer that already has ``qr.png`` open never observes a
    half-written file while a token is being refreshed.

    Args:
        url: Login URL returned by Telethon's ``QRLogin.url``.
        path: Destination file; defaults to :data:`QR_IMAGE_FILE` (resolved at
            call time so tests and callers can point it elsewhere).

    Raises:
        ConfigError: if the optional ``qrcode`` package (with its Pillow
            backend) is not installed.
        OSError: if the image cannot be written to disk.
    """
    target = path if path is not None else QR_IMAGE_FILE
    if qrcode is None:
        raise ConfigError(
            "The 'qrcode' package is required for --qr login. "
            'Install it with: pip install "qrcode[pil]"'
        )

    try:
        code = qrcode.QRCode(box_size=QR_IMAGE_BOX_SIZE, border=QR_IMAGE_BORDER)
        code.add_data(url)
        code.make(fit=True)
        image = code.make_image(fill_color="black", back_color="white")
    except ImportError as exc:  # Pillow missing => no PNG backend
        raise ConfigError(
            "Rendering qr.png requires Pillow. "
            'Install it with: pip install "qrcode[pil]"'
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(target)
    return target


def print_qr_instructions(path: Optional[Path] = None, refreshed: bool = False) -> None:
    """Tell the user where the QR image is and how to scan it."""
    target = path if path is not None else QR_IMAGE_FILE
    action = "updated" if refreshed else "generated"
    print()
    print(f"[QR] {target.name} {action}: {target}")
    print(
        "Open it in any image viewer and scan it with Telegram: "
        "Settings -> Devices -> Link Desktop Device."
    )
    print(
        f"The token expires in ~{int(QR_WAIT_TIMEOUT + 5)}s; "
        f"{target.name} is refreshed automatically while waiting."
    )


def cleanup_qr_image(path: Optional[Path] = None) -> None:
    """Delete the rendered QR image once it is no longer useful."""
    target = path if path is not None else QR_IMAGE_FILE
    try:
        target.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:  # pragma: no cover - only on locked/permission-denied files
        LOGGER.warning("Could not remove %s: %s", target, exc)
        return
    LOGGER.info("Removed %s (login complete).", target.name)


async def _complete_qr_password(
    client: TelegramClient, password: Optional[str]
) -> None:
    """Finish the 2FA challenge raised by the QR login flow.

    Raises:
        AuthError: if no password is available in a non-interactive shell or
            the supplied password is rejected.
    """
    LOGGER.info("Two-step verification is enabled.")
    if password:
        await submit_password(client, password)
    elif sys.stdin.isatty():
        await sign_in_with_password(client)
    else:
        raise AuthError(
            "Two-step verification is enabled but no --password was given. "
            "Re-run with --qr --password <PASSWORD> to finish."
        )


async def qr_authenticate(
    client: TelegramClient, password: Optional[str] = None
) -> None:
    """Link the account by rendering a QR login token to ``qr.png``.

    The token is awaited in a loop: when Telegram expires it (``wait`` raises
    :class:`asyncio.TimeoutError`), a fresh one is requested with
    ``qr_login.recreate()`` and ``qr.png`` is overwritten so the user always
    scans the currently valid token.

    Args:
        client: Connected (but unauthorized) Telethon client.
        password: Cloud password, if two-step verification is enabled and the
            shell is non-interactive.

    Raises:
        ConfigError: if the ``qrcode`` renderer is unavailable.
        AuthError: if the 2FA challenge cannot be completed.
        OSError: if ``qr.png`` cannot be written.
    """
    qr = await client.qr_login()
    rendered_url: Optional[str] = None
    refreshed = False

    while not await client.is_user_authorized():
        if qr.url != rendered_url:
            await asyncio.to_thread(render_qr, qr.url)
            print_qr_instructions(refreshed=refreshed)
            rendered_url = qr.url
            refreshed = True

        try:
            await qr.wait(timeout=QR_WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            LOGGER.info("QR token expired; generating a new one.")
            await qr.recreate()
            continue
        except SessionPasswordNeededError:
            await _complete_qr_password(client, password)
            break

    if not await client.is_user_authorized():
        raise AuthError(
            "Telegram still reports the client as unauthorized after QR login."
        )


async def authenticate(
    phone: Optional[str] = None,
    fresh: bool = False,
    code: Optional[str] = None,
    password: Optional[str] = None,
    request_only: bool = False,
    qr: bool = False,
) -> int:
    """Authorize the session via phone + code or a QR token and verify the file.

    When ``request_only`` is true the login code is requested and the pending
    state is persisted, but nothing is signed in: the run exits with
    :data:`EXIT_OK` so the code can be submitted by a later ``--code`` run.

    When ``qr`` is true the account is linked through ``qr.png`` instead: no
    phone number is requested and the login loop lives in :func:`qr_authenticate`.
    """
    await load_environment()
    api_id, api_hash = read_api_credentials()
    session_name = read_session_name()
    proxy = build_proxy()

    if fresh:
        await asyncio.to_thread(reset_session_files, session_name)

    client = TelegramClient(session_name, api_id, api_hash, proxy=proxy)
    me = None
    try:
        LOGGER.info("Connecting to Telegram (session=%s)...", session_name)
        await client.connect()

        if await client.is_user_authorized():
            LOGGER.info("Existing session is already authorized — skipping login.")
        elif qr:
            LOGGER.info("Not authorized — starting QR login.")
            await qr_authenticate(client, password=password)
        else:
            # Only ask for the number when a real login is needed, so an
            # already-authorized session stays non-interactive.
            pending_phone = load_pending_state().get("phone")
            target_phone = await resolve_phone(phone or pending_phone)
            LOGGER.info(
                "Not authorized — starting phone login for %s.",
                mask_phone(target_phone),
            )
            if request_only:
                sent = await request_login_code(client, target_phone)
                save_pending_state(target_phone, sent.phone_code_hash)
                print()
                print("[PENDING] Login code requested.")
                print(f"Phone: {mask_phone(target_phone)}")
                print(
                    "Next: python auth_test.py --code <CODE>"
                    "  (append --password <PASSWORD> if 2FA is enabled)"
                )
                return EXIT_OK
            await phone_authenticate(
                client, target_phone, code=code, password=password
            )

        me = await client.get_me()
        if me is None:
            LOGGER.error("Login completed but no user profile was returned.")
            return EXIT_AUTH_ERROR
    finally:
        if client.is_connected():
            await client.disconnect()
            LOGGER.info("Disconnected from Telegram.")

    session_path = verify_session_file(session_name)
    stat = session_path.stat()
    clear_pending_state()
    await asyncio.to_thread(cleanup_qr_image)

    full_name = " ".join(
        part for part in (me.first_name, me.last_name) if part
    ) or "(not set)"
    username = f"@{me.username}" if me.username else "(not set)"

    print()
    print("[SUCCESS] Authenticated successfully!")
    print(f"Name: {full_name}")
    print(f"Username: {username}")
    print(f"User ID: {me.id}")
    print(f"Session saved: {session_path} ({stat.st_size} bytes)")
    return EXIT_OK


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI flags for the interactive login."""
    parser = argparse.ArgumentParser(
        description=(
            "Authorize the Telethon session with a phone number and the login "
            "code Telegram sends to your device, or by scanning a QR token "
            "rendered to qr.png (--qr)."
        )
    )
    parser.add_argument(
        "--phone",
        default=None,
        metavar="+7XXXXXXXXXX",
        help=(
            "Phone number in international format. Overrides TG_PHONE; when "
            "neither is given the number is prompted for interactively."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Back up and discard any existing session file before logging in "
            "(writes <session>.session.stale.bak)."
        ),
    )
    parser.add_argument(
        "--code",
        default=None,
        metavar="CODE",
        help=(
            "Login code from Telegram. Submits it against the pending request "
            "saved by the previous run instead of requesting a new code."
        ),
    )
    parser.add_argument(
        "--password",
        default=None,
        metavar="PASSWORD",
        help="Two-step verification cloud password (only if 2FA is enabled).",
    )
    parser.add_argument(
        "--request-code",
        action="store_true",
        help=(
            "Only request the login code and save the pending state, then "
            "exit. Use in non-interactive shells; submit the code with --code."
        ),
    )
    parser.add_argument(
        "--qr",
        action="store_true",
        help=(
            "Log in by scanning a QR token instead of a phone code. The token "
            "is written to qr.png in the project root and refreshed on expiry; "
            "scan it via Telegram -> Settings -> Devices -> Link Desktop Device."
        ),
    )
    args = parser.parse_args(argv)
    if args.qr and (args.code or args.request_code or args.phone):
        parser.error(
            "--qr cannot be combined with --phone, --code or --request-code."
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Synchronous CLI wrapper translating failures into exit codes."""
    configure_logging()
    args = parse_args(argv)
    try:
        return asyncio.run(
            authenticate(
                phone=args.phone,
                fresh=args.fresh,
                code=args.code,
                password=args.password,
                request_only=args.request_code,
                qr=args.qr,
            )
        )
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return EXIT_CONFIG_ERROR
    except SessionError as exc:
        LOGGER.error("%s", exc)
        return EXIT_SESSION_ERROR
    except AuthError as exc:
        LOGGER.error("Authentication failed: %s", exc)
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
