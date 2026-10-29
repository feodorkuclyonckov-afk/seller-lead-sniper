"""Safely join curated public B2B/e-commerce Telegram chats.

Purpose
-------
Bootstrap the ``TG_MONITOR_CHATS`` list used by ``python main.py --live`` by
joining a small, curated set of public B2B chats (cargo, e-commerce,
fulfillment, marketplaces, marketing). The script is deliberately conservative:

* it reuses the authorized Telethon session created by ``auth_test.py``;
* it never crashes on a single bad/private/renamed chat;
* it honours ``FloodWaitError`` and backs off;
* it sleeps 45-90 seconds between successful joins to avoid spam triggers;
* it merges successfully resolved chats into the ``TG_MONITOR_CHATS`` line in
  ``.env`` without dropping entries that are already monitored.

Usage
-----
1. Run ``python auth_test.py`` once to authorize the session.
2. Run ``python join_chats.py`` (joins at most ``--limit`` chats, default 10).
3. Start the listener with ``python main.py --live``.

Useful flags
------------
``--dry-run``  Resolve the target chats without joining and leave ``.env``
               untouched.
``--limit N``  Process at most ``N`` chats in this batch (default 10). A value
               of ``0`` exits immediately without touching Telegram.
``--discover`` Search Telegram's public directory by keyword, keep joinable
               discussion groups/supergroups and feed them into the join
               pipeline (alias: ``--auto-search``). Custom keywords may be
               passed with ``--query "text"`` (repeatable); otherwise
               ``keywords.txt`` is used when present, falling back to the
               built-in niche query list.

Exit codes
----------
0  At least one chat was joined/verified and ``.env`` was updated.
1  Configuration problem (missing/invalid ``.env`` values).
2  The Telegram session is not authorized.
3  No chats were joined/verified; ``.env`` was left unchanged.
4  Unexpected runtime failure.
"""

from __future__ import annotations

import argparse
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
    ChannelsTooMuchError,
    FloodWaitError,
    UserAlreadyParticipantError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.contacts import SearchRequest

try:  # python-socks is only required when a proxy is actually configured.
    import python_socks  # noqa: F401
except ImportError:  # pragma: no cover - exercised only on broken installs
    python_socks = None  # type: ignore[assignment]

LOGGER = logging.getLogger("lead_sniper.join_chats")

ENV_FILE: Path = Path(__file__).resolve().with_name(".env")
TARGETS_FILE: Path = Path(__file__).resolve().with_name("targets.txt")
KEYWORDS_FILE: Path = Path(__file__).resolve().with_name("keywords.txt")
DEFAULT_SESSION_NAME: str = "session_auth_test"
MONITOR_CHATS_ENV: str = "TG_MONITOR_CHATS"

#: Curated public B2B/e-commerce chats. Usernames are stored without ``@``.
#: Edit this tuple to target a different niche. The list is intentionally
#: ordered so the most relevant marketplace/cargo/fulfillment chats are
#: processed first when a small ``--limit`` is used.
TARGET_CHATS: Tuple[str, ...] = (
    "wb_chat_sellers",
    "fulfillment_russia",
    "cargo_china_chat",
    "chat_wildberries_ozon",
    "sellers_wildberries",
    "market_place_ru",
    "wb_sellers_club",
    "marketplace_help",
    "chatozonwb",
    "ff_sklad_wb",
    "fulfillment_moscow",
    "chinacargo_chat",
    "dostavka_iz_kitaya_cargo",
    "wb_managers",
    "infografika_wb_ozon",
    # Previously curated chats kept so the monitored list only ever grows.
    "marketplace_chat_wb_ozon",
    "cargo_chat_moscow",
    "ecom_russia",
)

#: Built-in niche search queries used by ``--discover`` when neither
#: ``--query`` nor a ``keywords.txt`` file is supplied. Each query is sent to
#: Telegram's global directory (``contacts.SearchRequest``).
DEFAULT_SEARCH_QUERIES: Tuple[str, ...] = (
    "wildberries карго",
    "ozon фулфилмент",
    "селлеры wildberries",
    "доставка китай карго",
    "вайлдберриз поставщики",
)

#: Maximum number of directory hits requested per search query.
SEARCH_RESULT_LIMIT: int = 20

#: Matches a Telegram chat URL prefix (``https://t.me/``, ``t.me/``,
#: ``https://telegram.me/``) so links can be normalised to bare usernames.
_TARGET_URL_PREFIX = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/",
    re.IGNORECASE,
)


def normalize_target(raw: str) -> str:
    """Return a bare username from a raw target line.

    Accepts plain usernames, ``@username`` handles and public chat links such
    as ``https://t.me/username`` (optionally with a message id or query
    string). Empty or whitespace-only input yields an empty string.
    """
    value = raw.strip()
    if not value:
        return ""
    value = _TARGET_URL_PREFIX.sub("", value)
    value = value.lstrip("@")
    # Drop any trailing path segment (message id) or query string.
    value = value.split("?", 1)[0].split("/", 1)[0]
    return value.strip()


def load_targets(path: Optional[Path] = None) -> list[str]:
    """Load target chats from ``targets.txt``.

    Blank lines and lines starting with ``#`` are ignored. Each entry is
    normalised with :func:`normalize_target` and duplicates are removed while
    preserving first-seen order. When the file does not exist, the built-in
    :data:`TARGET_CHATS` pool is returned so the script keeps working.
    """
    target_path = path if path is not None else TARGETS_FILE
    if not target_path.is_file():
        LOGGER.info(
            "Targets file %s not found; falling back to built-in TARGET_CHATS (%d).",
            target_path,
            len(TARGET_CHATS),
        )
        return list(TARGET_CHATS)

    targets: list[str] = []
    for line in target_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = normalize_target(stripped)
        if name and name not in targets:
            targets.append(name)

    LOGGER.info("Loaded %d target chat(s) from %s.", len(targets), target_path)
    return targets


def load_queries(path: Optional[Path] = None) -> list[str]:
    """Load search queries from ``keywords.txt``.

    Blank lines and lines starting with ``#`` are ignored; duplicates are
    removed while preserving first-seen order. When the file does not exist,
    the built-in :data:`DEFAULT_SEARCH_QUERIES` list is returned so
    ``--discover`` works out of the box.
    """
    keywords_path = path if path is not None else KEYWORDS_FILE
    if not keywords_path.is_file():
        LOGGER.info(
            "Keywords file %s not found; using built-in search queries (%d).",
            keywords_path,
            len(DEFAULT_SEARCH_QUERIES),
        )
        return list(DEFAULT_SEARCH_QUERIES)

    queries: list[str] = []
    for line in keywords_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped not in queries:
            queries.append(stripped)

    LOGGER.info("Loaded %d search quer(y/ies) from %s.", len(queries), keywords_path)
    return queries


def resolve_queries(
    cli_queries: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
) -> list[str]:
    """Return the effective search queries for ``--discover``.

    Explicit ``--query`` values take precedence; otherwise ``keywords.txt`` is
    used when present, and the built-in list is the final fallback.
    """
    if cli_queries:
        queries: list[str] = []
        for raw in cli_queries:
            query = (raw or "").strip()
            if query and query not in queries:
                queries.append(query)
        if queries:
            return queries
    return load_queries(path)


#: Delay range (seconds) between successful joins. Kept deliberately wide to
#: stay well below Telegram's anti-spam thresholds.
MIN_JOIN_DELAY: float = 45.0
MAX_JOIN_DELAY: float = 90.0

#: A flood wait longer than this is treated as "skip this chat".
MAX_FLOOD_WAIT: int = 300

#: How many times a single chat is attempted after a tolerable flood wait.
MAX_JOIN_ATTEMPTS: int = 2

OUTCOME_JOINED: str = "joined"
OUTCOME_ALREADY: str = "already"
OUTCOME_RESOLVED: str = "resolved"
OUTCOME_SKIPPED: str = "skipped"

#: Default number of chats processed per execution batch.
DEFAULT_LIMIT: int = 10

EXIT_OK: int = 0
EXIT_CONFIG_ERROR: int = 1
EXIT_AUTH_ERROR: int = 2
EXIT_NO_CHATS: int = 3
EXIT_RUNTIME_ERROR: int = 4

# Hosts that mean "no proxy" — a local SOCKS listener is not a real egress.
_DIRECT_HOSTS: frozenset[str] = frozenset({"", "127.0.0.1", "localhost"})

#: Proxy schemes understood by python-socks / Telethon.
_SUPPORTED_PROXY_TYPES: frozenset[str] = frozenset({"socks5", "socks4", "http"})

#: Default scheme. The configured provider only speaks HTTP CONNECT on the
#: port it advertises as SOCKS5, so ``http`` is the working default.
DEFAULT_PROXY_TYPE: str = "http"

# Telethon proxy config: a python-socks style dict, or None for direct.
ProxyConfig = Optional[dict]


class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


class AuthError(RuntimeError):
    """Raised when the Telethon session is not authorized."""


class AccountLimitReached(RuntimeError):
    """Raised when Telegram reports the account has joined too many channels."""


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
    """Load ``.env`` so it is the single source of truth for configuration.

    ``override`` is ``True`` so the project ``.env`` wins over variables
    exported by the shell (or inherited from a previous run): a stale
    ``PROXY_HOST`` must not silently re-enable a proxy that the file
    explicitly leaves empty.
    """
    if not ENV_FILE.is_file():
        raise ConfigError(
            f"Environment file not found: {ENV_FILE}. "
            "Copy .env.example to .env and fill in your credentials."
        )
    load_dotenv(dotenv_path=ENV_FILE, override=True)


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

    ``rdns`` is deliberately ``False`` so Telethon connects to Telegram's
    numeric DC IP addresses directly through the proxy; remote DNS resolution
    (``rdns=True``) times out on this provider.
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


async def _join_one(
    client: TelegramClient, username: str, dry_run: bool = False
) -> str:
    """Attempt to join ``username`` and return an outcome constant.

    ``client.get_entity`` resolves the public username first; the actual join
    is performed with ``JoinChannelRequest``. Existing membership surfaces as
    ``UserAlreadyParticipantError``, which is the canonical "already joined"
    signal in Telethon.

    When ``dry_run`` is true the entity is resolved but no join request is
    sent, and ``OUTCOME_RESOLVED`` is returned.

    Raises:
        AccountLimitReached: if Telegram reports that the account has joined
            the maximum number of channels.
    """
    for attempt in range(1, MAX_JOIN_ATTEMPTS + 1):
        try:
            entity = await client.get_entity(username)
            if dry_run:
                return OUTCOME_RESOLVED
            await client(JoinChannelRequest(entity))
            return OUTCOME_JOINED
        except UserAlreadyParticipantError:
            return OUTCOME_ALREADY
        except ChannelsTooMuchError as exc:
            raise AccountLimitReached(str(exc)) from exc
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


def _is_joinable_chat(chat: object) -> bool:
    """Return ``True`` when ``chat`` is a public, joinable discussion group.

    The lead bot needs groups where clients can post requests, so broadcast-only
    channels are rejected: a channel must either be a megagroup or explicitly
    expose comments. Chats without a public ``username`` cannot be joined by
    link and are rejected, as are chats the account has already left (i.e. it
    is not currently a member).
    """
    username = getattr(chat, "username", None)
    if not isinstance(username, str) or not username.strip():
        return False
    # ``left is False`` means the account is already a member: nothing to join.
    if getattr(chat, "left", None) is False:
        return False
    if bool(getattr(chat, "deactivated", False)):
        return False
    if bool(getattr(chat, "comments_enabled", False)):
        return True
    # Basic groups (Chat) expose no broadcast/megagroup flags and are joinable.
    if not (hasattr(chat, "megagroup") or hasattr(chat, "broadcast")):
        return True
    if bool(getattr(chat, "broadcast", False)):
        return False
    return bool(getattr(chat, "megagroup", False))


async def search_target_chats(
    client: TelegramClient,
    queries: Sequence[str],
    limit: int = SEARCH_RESULT_LIMIT,
) -> list[str]:
    """Search Telegram's global directory and return joinable usernames.

    Each query is sent through ``contacts.SearchRequest`` with a per-query
    result cap of ``limit``. Only public, joinable discussion groups are kept
    (see :func:`_is_joinable_chat`); usernames are normalised and deduplicated
    while preserving discovery order. A failing or flood-limited query is
    logged and skipped so one bad keyword cannot abort the run.
    """
    per_query_limit = max(int(limit), 1)
    discovered: list[str] = []

    for query in queries:
        text = (query or "").strip()
        if not text:
            continue
        LOGGER.info("Searching Telegram for %r (limit=%d)...", text, per_query_limit)

        result = None
        for attempt in range(1, MAX_JOIN_ATTEMPTS + 1):
            try:
                result = await client(SearchRequest(q=text, limit=per_query_limit))
                break
            except FloodWaitError as exc:
                seconds = int(getattr(exc, "seconds", 0) or 0)
                if seconds > MAX_FLOOD_WAIT or attempt >= MAX_JOIN_ATTEMPTS:
                    LOGGER.warning(
                        "Flood wait of %ss while searching %r; skipping query.",
                        seconds,
                        text,
                    )
                    result = None
                    break
                LOGGER.warning(
                    "Flood wait: sleeping %ss before retrying search %r (attempt %d/%d).",
                    seconds,
                    text,
                    attempt,
                    MAX_JOIN_ATTEMPTS,
                )
                await asyncio.sleep(seconds)
            except Exception as exc:  # noqa: BLE001 - one bad query must not abort the run
                LOGGER.warning("Search for %r failed: %s", text, exc)
                result = None
                break

        if result is None:
            continue

        for chat in getattr(result, "chats", None) or []:
            if not _is_joinable_chat(chat):
                continue
            username = normalize_target(str(getattr(chat, "username", "") or ""))
            if not username or username in discovered:
                continue
            discovered.append(username)
            LOGGER.info(
                "Discovered chat: %s (@%s) via query %r.",
                getattr(chat, "title", None) or "(no title)",
                username,
                text,
            )

    LOGGER.info(
        "Discovered %d joinable chat(s) across %d search quer(y/ies).",
        len(discovered),
        len(queries),
    )
    return discovered


async def join_targets(
    client: TelegramClient,
    limit: int,
    dry_run: bool = False,
    targets: Optional[Sequence[str]] = None,
) -> Tuple[list[str], list[str]]:
    """Join up to ``limit`` target chats, returning ``(resolved, skipped)``.

    ``resolved`` contains chats that were joined, were already joined, or (in
    dry-run mode) resolved successfully. Processing stops early if Telegram
    reports that the account has reached its channel limit.

    ``targets`` is the ordered pool to process; when omitted it is loaded from
    ``targets.txt`` (falling back to :data:`TARGET_CHATS`).
    """
    joined: list[str] = []
    skipped: list[str] = []
    pool = list(targets) if targets is not None else load_targets()
    selected = pool[: max(limit, 0)]
    total = len(selected)

    for index, username in enumerate(selected, start=1):
        LOGGER.info("Processing %d/%d: %s", index, total, username)
        try:
            outcome = await _join_one(client, username, dry_run=dry_run)
        except AccountLimitReached as exc:
            LOGGER.error(
                "Account channel limit reached (%s); stopping after %d chat(s).",
                exc,
                len(joined),
            )
            print("[-] Account channel limit reached; stopping.")
            break

        if outcome == OUTCOME_JOINED:
            print(f"[+] Successfully joined: {username}")
            joined.append(username)
        elif outcome == OUTCOME_ALREADY:
            print(f"[*] Already member: {username}")
            joined.append(username)
        elif outcome == OUTCOME_RESOLVED:
            print(f"[*] Resolved (dry-run): {username}")
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


def _read_monitor_chats(env_path: Path) -> list[str]:
    """Return the current ``TG_MONITOR_CHATS`` entries from ``env_path``."""
    if not env_path.is_file():
        return []
    pattern = re.compile(
        rf"^\s*(?:export\s+)?{re.escape(MONITOR_CHATS_ENV)}\s*=(.*)$"
    )
    for line in env_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            raw = match.group(1).strip().strip('"').strip("'")
            return [item.strip() for item in raw.split(",") if item.strip()]
    return []


def update_env_file(env_path: Path, usernames: Sequence[str]) -> str:
    """Merge ``usernames`` into ``TG_MONITOR_CHATS`` in ``env_path``.

    Existing entries are preserved so the monitored list only ever grows;
    duplicates are removed while keeping first-seen order. Existing variables,
    comments and line endings are preserved; only the ``TG_MONITOR_CHATS``
    line is touched. Returns the new value.
    """
    merged: list[str] = []
    for name in [*_read_monitor_chats(env_path), *usernames]:
        if name not in merged:
            merged.append(name)
    value = ",".join(merged)
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


def append_targets(
    usernames: Sequence[str],
    path: Optional[Path] = None,
) -> list[str]:
    """Persist freshly discovered usernames in ``targets.txt``.

    Existing entries, comments and line endings are preserved; only genuinely
    new usernames are appended. Returns the list of usernames actually added
    (empty when everything was already present).
    """
    target_path = path if path is not None else TARGETS_FILE

    existing: list[str] = []
    text = ""
    if target_path.is_file():
        text = target_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = normalize_target(stripped)
            if name and name not in existing:
                existing.append(name)

    added: list[str] = []
    for raw in usernames:
        name = normalize_target(raw)
        if name and name not in existing and name not in added:
            added.append(name)

    if not added:
        return []

    newline = "\r\n" if "\r\n" in text else "\n"
    if text and not text.endswith(("\n", "\r")):
        text += newline
    if "# Discovered via join_chats.py --discover" not in text:
        text += "# Discovered via join_chats.py --discover" + newline
    text += "".join(name + newline for name in added)
    target_path.write_text(text, encoding="utf-8", newline="")

    LOGGER.info("Appended %d new target(s) to %s.", len(added), target_path)
    return added


async def run(args: argparse.Namespace) -> int:
    """Connect, join the target chats and persist the result to ``.env``."""
    load_environment()

    discover = bool(getattr(args, "discover", False))
    queries = resolve_queries(getattr(args, "query", None)) if discover else []
    targets = load_targets()
    LOGGER.info("Loaded %d target chat(s) to process.", len(targets))
    if discover:
        LOGGER.info(
            "Automatic discovery enabled with %d search quer(y/ies).", len(queries)
        )

    if args.limit <= 0:
        LOGGER.info("--limit 0: no chats to process.")
        print("[*] Limit is 0; nothing to do.")
        return EXIT_OK

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

        if discover:
            discovered = await search_target_chats(client, queries)
            if discovered:
                print(
                    f"[*] Discovered {len(discovered)} joinable chat(s): "
                    f"{', '.join(discovered)}"
                )
            else:
                print("[-] Automatic search found no joinable chats.")
            targets = discovered
            if not args.dry_run and discovered:
                added = append_targets(discovered)
                if added:
                    print(
                        f"[*] Added {len(added)} new target(s) to "
                        f"{TARGETS_FILE.name}"
                    )

        joined, skipped = await join_targets(
            client, limit=args.limit, dry_run=args.dry_run, targets=targets
        )
    finally:
        if client.is_connected():
            await client.disconnect()
            LOGGER.info("Disconnected from Telegram.")

    if args.dry_run:
        print()
        print("[*] Dry-run complete; .env was not modified.")
        if skipped:
            print(f"[*] Skipped {len(skipped)} chat(s): {', '.join(skipped)}")
        return EXIT_OK

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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI flags for a controlled join batch."""
    parser = argparse.ArgumentParser(
        description=(
            "Safely join curated B2B/e-commerce Telegram chats and merge the "
            "resolved usernames into TG_MONITOR_CHATS in .env."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve target chats without joining and leave .env untouched.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=(
            "Maximum number of chats to process in this batch "
            f"(default: {DEFAULT_LIMIT}; 0 exits immediately)."
        ),
    )
    parser.add_argument(
        "--discover",
        "--auto-search",
        dest="discover",
        action="store_true",
        help=(
            "Search Telegram's public directory by keyword, keep joinable "
            "discussion groups and feed them into the join pipeline."
        ),
    )
    parser.add_argument(
        "--query",
        action="append",
        metavar="TEXT",
        default=None,
        help=(
            "Search query used by --discover; repeatable. Defaults to "
            f"{KEYWORDS_FILE.name} when present, else the built-in niche list."
        ),
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
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
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI entry point
        LOGGER.error("Unexpected failure: %s", exc)
        LOGGER.debug("Traceback:", exc_info=True)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
