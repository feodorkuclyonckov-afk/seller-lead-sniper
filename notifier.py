"""Sprint 4 — async Telegram alert notifier for qualified leads.

Purpose
-------
Sprint 3's :mod:`classifier` decides *whether* a scraped message is a lead and
labels it with a category, a short summary and a confidence score. The final
step of the pipeline is to get that lead in front of a human quickly. This
module formats a compact, readable alert and delivers it through the Telegram
Bot API.

The notifier is deliberately defensive, mirroring :mod:`classifier`: a scraping
pipeline must never die because Telegram rate-limited us, a network request
timed out, or the bot token was left blank. Every failure path returns ``False``
instead of raising.

Public API
----------
``TelegramConfig``
    Immutable settings container (``bot_token``, ``chat_id``, ``timeout``).
``read_telegram_config()``
    Synchronous reader for the current process environment.
``resolve_alert_chat_id(category)``
    Resolve the destination chat id from the environment, preferring a
    category-specific route over the default chat.
``load_telegram_config(env_file)``
    Asynchronously load ``.env`` and return a :class:`TelegramConfig`.
``format_alert(lead_data, original_text, source_chat, msg_link)``
    Render the HTML message body sent to Telegram.
``send_lead_alert(lead_data, original_text, source_chat, msg_link)``
    Format and deliver an alert, retrying transient failures. Returns ``True``
    on success and ``False`` on any failure.

Message format
--------------
Alerts use Telegram's ``HTML`` parse mode (far less error-prone than
MarkdownV2, which requires escaping a dozen punctuation characters)::

    🎯 [Карго / Логистика] — Рейтинг: 9/10

    Поиск карго из Китая в Москву

    📍 Источник: Карго Чат Москва
    🔗 Открыть сообщение

    «Ищу карго из Китая в Москву, кто занимается доставкой под ключ?»

All user-controlled fragments are HTML-escaped, and the final body is capped at
Telegram's 4096-character limit.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote, urlparse

import httpx
from dotenv import load_dotenv

LOGGER = logging.getLogger("lead_sniper.notifier")

# --------------------------------------------------------------------------- #
# Configuration defaults
# --------------------------------------------------------------------------- #

#: Root of the Telegram Bot API. The bot token is appended per request.
API_BASE: Final[str] = "https://api.telegram.org"

#: Per-request timeout in seconds. Kept short so a stalled API cannot block the
#: pipeline for long.
DEFAULT_TIMEOUT: Final[float] = 15.0

#: Number of *additional* attempts for transient failures (timeouts, connection
#: resets, HTTP 429 and 5xx).
DEFAULT_MAX_RETRIES: Final[int] = 2

#: Base delay for exponential backoff between retries, in seconds.
RETRY_BASE_DELAY: Final[float] = 1.0

#: Hard cap on any single backoff / ``retry_after`` delay, in seconds.
MAX_RETRY_DELAY: Final[float] = 30.0

#: Telegram's hard limit for a single message body, in characters.
MAX_MESSAGE_LENGTH: Final[int] = 4096

#: Maximum number of characters kept from the original message snippet.
MAX_SNIPPET_CHARS: Final[int] = 600

#: Maximum number of characters kept from the classifier summary.
MAX_SUMMARY_CHARS: Final[int] = 200

#: Maximum number of characters kept from the source chat name.
MAX_SOURCE_CHARS: Final[int] = 100

#: Category used when the classifier payload omits one.
DEFAULT_CATEGORY: Final[str] = "Другое"

#: Inclusive bounds for the confidence score.
MIN_CONFIDENCE: Final[int] = 1
MAX_CONFIDENCE: Final[int] = 10

#: Telegram parse mode used for every alert.
PARSE_MODE: Final[str] = "HTML"

#: Location of the project ``.env`` file (next to this module).
ENV_FILE: Final[Path] = Path(__file__).resolve().with_name(".env")

#: Proxy environment variables shared with the Telethon client.
PROXY_TYPE_ENV: Final[str] = "PROXY_TYPE"
PROXY_HOST_ENV: Final[str] = "PROXY_HOST"
PROXY_PORT_ENV: Final[str] = "PROXY_PORT"
PROXY_USER_ENV: Final[str] = "PROXY_USER"
PROXY_PASS_ENV: Final[str] = "PROXY_PASS"

#: Default proxy scheme. The configured provider only speaks HTTP CONNECT on
#: the port it advertises as SOCKS5, so ``http`` is the working default.
DEFAULT_PROXY_TYPE: Final[str] = "http"

#: Proxy schemes accepted by :func:`resolve_proxy_url`.
SUPPORTED_PROXY_TYPES: Final[frozenset[str]] = frozenset(
    {"http", "https", "socks5", "socks4"}
)

#: Hosts that mean "no proxy" — a local listener is not a real egress.
DIRECT_PROXY_HOSTS: Final[frozenset[str]] = frozenset(
    {"", "127.0.0.1", "localhost"}
)

#: Primary environment variable holding the default destination chat id.
DEFAULT_CHAT_ENV: Final[str] = "TELEGRAM_ALERT_CHAT_ID"

#: Secondary fallback used when :data:`DEFAULT_CHAT_ENV` is unset.
FALLBACK_CHAT_ENV: Final[str] = "ALERT_CHAT_DEFAULT"

#: Category (lower-case) -> dedicated destination chat environment variable.
#:
#: Both the English keys and the exact Russian labels emitted by
#: :data:`classifier.CATEGORIES` are accepted, so a classifier payload routes
#: correctly without any translation layer. Keys are matched after stripping
#: surrounding whitespace and lower-casing (see :func:`resolve_alert_chat_id`).
CATEGORY_CHAT_ENV: Final[dict[str, str]] = {
    # English aliases.
    "fulfillment": "ALERT_CHAT_FULFILLMENT",
    "cargo": "ALERT_CHAT_CARGO",
    "marketing": "ALERT_CHAT_MARKETING",
    "design": "ALERT_CHAT_DESIGN",
    # Russian labels emitted by classifier.CATEGORIES.
    "фулфилмент": "ALERT_CHAT_FULFILLMENT",
    "карго / логистика": "ALERT_CHAT_CARGO",
    "маркетинг / трафик": "ALERT_CHAT_MARKETING",
    "дизайн / контент": "ALERT_CHAT_DESIGN",
}

#: Matches a trailing, possibly incomplete HTML entity (``&amp``, ``&#12``).
_TRAILING_ENTITY_RE: Final[re.Pattern[str]] = re.compile(r"&[a-zA-Z#0-9]*$")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Immutable Telegram settings with safe defaults.

    Attributes:
        bot_token: Bot API token from ``@BotFather``. Empty means "disabled".
        chat_id: Target chat — a numeric id (``-1001234567890``) or a public
            channel username (``@my_channel``). Empty means "disabled".
        timeout: Per-request timeout in seconds.
    """

    bot_token: str = ""
    chat_id: str = ""
    timeout: float = DEFAULT_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """``True`` when both a token and a chat id are present."""
        return bool(self.bot_token and self.chat_id)


def _read(name: str, default: str = "") -> str:
    """Return a stripped environment value, falling back to ``default``."""
    value = (os.getenv(name) or "").strip()
    return value or default


def _normalise_chat_id(raw: str) -> str | None:
    """Validate and normalise a Telegram chat id.

    Accepts a numeric id (optionally negative, as used by supergroups) or a
    public channel username prefixed with ``@``. Anything else is rejected so
    that a typo surfaces as "not configured" instead of a confusing HTTP 400.

    Returns:
        The cleaned chat id, or ``None`` when ``raw`` is blank or invalid.
    """
    value = (raw or "").strip()
    if not value:
        return None

    if value.startswith("@"):
        return value if len(value) > 1 else None

    try:
        int(value)
    except ValueError:
        return None
    return value


def resolve_alert_chat_id(category: str | None = None) -> str:
    """Resolve the destination Telegram chat id from the environment.

    A recognised ``category`` (see :data:`CATEGORY_CHAT_ENV`) whose dedicated
    environment variable holds a value wins. Every other case — unknown or
    missing category, or a blank category variable — falls back to
    ``TELEGRAM_ALERT_CHAT_ID`` and then ``ALERT_CHAT_DEFAULT``.

    Args:
        category: Category label from the classifier payload. Matched
            case-insensitively after stripping surrounding whitespace.

    Returns:
        The non-empty chat id to deliver the alert to.

    Raises:
        RuntimeError: When neither a category route nor any fallback chat id
            is configured.
    """
    normalised = (category or "").strip().lower()
    env_name = CATEGORY_CHAT_ENV.get(normalised) if normalised else None

    if env_name is not None:
        routed = _read(env_name)
        if routed:
            return routed
        LOGGER.debug(
            "%s is empty for category %r; falling back to the default chat.",
            env_name,
            category,
        )

    fallback = _read(DEFAULT_CHAT_ENV) or _read(FALLBACK_CHAT_ENV)
    if fallback:
        return fallback

    raise RuntimeError(
        "No destination Telegram chat id is configured — set "
        f"{DEFAULT_CHAT_ENV}, {FALLBACK_CHAT_ENV}, or a category variable "
        f"({', '.join(sorted(set(CATEGORY_CHAT_ENV.values())))}) in the "
        "environment."
    )


def read_telegram_config() -> TelegramConfig:
    """Build a :class:`TelegramConfig` from the current process environment.

    Missing or blank variables leave the notifier disabled. An invalid chat id
    is logged and treated as missing. This function never raises.
    """
    token = _read("TELEGRAM_BOT_TOKEN")
    raw_chat = _read("TELEGRAM_ALERT_CHAT_ID")
    chat_id = _normalise_chat_id(raw_chat)

    if raw_chat and chat_id is None:
        LOGGER.warning(
            "TELEGRAM_ALERT_CHAT_ID=%r is not a numeric id or @username — "
            "alerting stays disabled.",
            raw_chat,
        )

    return TelegramConfig(bot_token=token, chat_id=chat_id or "")


def resolve_proxy_url() -> str | None:
    """Build an ``httpx`` proxy URL from the ``PROXY_*`` environment variables.

    The notifier shares the same proxy settings as the Telethon client, so an
    alert is delivered over the same egress path that already works for the
    scraper. A proxy is used only when ``PROXY_HOST`` is a non-empty string
    (after stripping whitespace) that is not a loopback address.

    Returns:
        A standard proxy URL such as ``http://user:pass@host:port``, or
        ``None`` when no proxy is configured or the configuration is invalid.
        This function never raises: a malformed value is logged and treated as
        "connect directly".
    """
    host = _read(PROXY_HOST_ENV)
    if not host or host.lower() in DIRECT_PROXY_HOSTS:
        return None

    raw_port = _read(PROXY_PORT_ENV)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        LOGGER.warning(
            "%s=%r is not an integer — connecting directly.",
            PROXY_PORT_ENV,
            raw_port,
        )
        return None

    if not 0 < port < 65536:
        LOGGER.warning(
            "%s=%d is out of range — connecting directly.",
            PROXY_PORT_ENV,
            port,
        )
        return None

    scheme = (_read(PROXY_TYPE_ENV) or DEFAULT_PROXY_TYPE).lower()
    if scheme not in SUPPORTED_PROXY_TYPES:
        LOGGER.warning(
            "%s=%r is unsupported — connecting directly.",
            PROXY_TYPE_ENV,
            scheme,
        )
        return None

    user = _read(PROXY_USER_ENV)
    password = _read(PROXY_PASS_ENV)
    auth = ""
    if user:
        auth = quote(user, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"

    return f"{scheme}://{auth}{host}:{port}"


async def load_telegram_config(
    env_file: str | Path | None = None,
) -> TelegramConfig:
    """Load ``.env`` off the event loop and return the resulting config.

    ``python-dotenv`` performs blocking file I/O, so it is dispatched to a
    worker thread. Existing process variables win (``override=False``), which
    keeps container/CI overrides working. Any error while reading the file is
    logged and ignored — the caller still receives usable defaults.

    Args:
        env_file: Optional path to the dotenv file. Defaults to the project
            ``.env`` next to this module.

    Returns:
        A :class:`TelegramConfig`; never raises.
    """
    path = Path(env_file) if env_file is not None else ENV_FILE

    if path.is_file():
        try:
            await asyncio.to_thread(
                load_dotenv, dotenv_path=path, override=False
            )
        except Exception:  # pragma: no cover - defensive, dotenv rarely fails
            LOGGER.warning("Could not read %s; using defaults.", path, exc_info=True)
    else:
        LOGGER.debug("No dotenv file at %s; using process environment.", path)

    return read_telegram_config()


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    """Escape ``&``, ``<`` and ``>`` for Telegram's HTML parse mode."""
    return html.escape(text, quote=False)


def _collapse(text: Any) -> str:
    """Flatten any value to a single line with collapsed whitespace."""
    return " ".join(str(text).split())


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` capped at ``limit`` characters, ending in an ellipsis."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


def _coerce_confidence(value: Any) -> int | None:
    """Coerce ``value`` to an int clamped to ``1..10``; ``None`` if invalid."""
    if isinstance(value, bool):
        return None
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, number))


def _safe_link(link: str | None) -> str | None:
    """Return a validated ``http(s)`` URL, or ``None`` when unusable."""
    if not link:
        return None
    candidate = link.strip()
    if not candidate:
        return None

    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        LOGGER.warning("Ignoring invalid message link: %r", link)
        return None
    return candidate


# --------------------------------------------------------------------------- #
# Alert formatting
# --------------------------------------------------------------------------- #


def format_alert(
    lead_data: dict,
    original_text: str,
    source_chat: str,
    msg_link: str | None = None,
) -> str:
    """Render a lead as a Telegram HTML message body.

    The output is safe to send with ``parse_mode="HTML"``: every interpolated
    fragment is escaped, the snippet is truncated, and the whole body is capped
    at :data:`MAX_MESSAGE_LENGTH`.

    Args:
        lead_data: Classifier output. Expected keys are ``category``,
            ``summary`` and ``confidence``; missing keys fall back to sensible
            defaults instead of raising.
        original_text: Raw message body, quoted as a snippet.
        source_chat: Human-readable name of the chat the lead came from.
        msg_link: Optional direct link to the original message.

    Returns:
        The formatted message body.
    """
    data: Mapping[str, Any] = (
        lead_data if isinstance(lead_data, Mapping) else {}
    )

    category = _collapse(data.get("category") or DEFAULT_CATEGORY) or DEFAULT_CATEGORY
    confidence = _coerce_confidence(data.get("confidence"))

    summary = _collapse(data.get("summary") or "")
    if not summary:
        summary = _truncate(_collapse(original_text or ""), 80) or "Без описания"
    summary = _truncate(summary, MAX_SUMMARY_CHARS)

    source = (
        _truncate(_collapse(source_chat or ""), MAX_SOURCE_CHARS)
        or "неизвестный чат"
    )

    snippet = _truncate(_collapse(original_text or ""), MAX_SNIPPET_CHARS) or "—"

    header = f"🎯 <b>[{_escape(category)}]</b>"
    if confidence is not None:
        header += f" — Рейтинг: {confidence}/10"

    summary_line = f"<b>{_escape(summary)}</b>"

    source_block = f"📍 Источник: <b>{_escape(source)}</b>"
    link = _safe_link(msg_link)
    if link is not None:
        source_block += (
            f'\n🔗 <a href="{html.escape(link, quote=True)}">'
            "Открыть сообщение</a>"
        )

    escaped_snippet = _escape(snippet)
    message = "\n\n".join(
        (
            header,
            summary_line,
            source_block,
            f"<blockquote>{escaped_snippet}</blockquote>",
        )
    )

    if len(message) <= MAX_MESSAGE_LENGTH:
        return message

    # The snippet is the only unbounded part; shrink it until the body fits.
    overhead = len(message) - len(escaped_snippet)
    budget = MAX_MESSAGE_LENGTH - overhead - 1  # reserve one char for "…"
    if budget < 0:
        budget = 0

    escaped_snippet = escaped_snippet[:budget]
    # Never leave a half-written entity such as "&am" at the cut point.
    escaped_snippet = _TRAILING_ENTITY_RE.sub("", escaped_snippet).rstrip()
    if escaped_snippet:
        escaped_snippet += "…"

    return "\n\n".join(
        (
            header,
            summary_line,
            source_block,
            f"<blockquote>{escaped_snippet}</blockquote>",
        )
    )


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


def _endpoint(bot_token: str) -> str:
    """Return the ``sendMessage`` endpoint for ``bot_token``."""
    return f"{API_BASE}/bot{bot_token}/sendMessage"


def _backoff(attempt: int) -> float:
    """Return the exponential backoff delay for a 1-based ``attempt``."""
    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
    return min(MAX_RETRY_DELAY, max(0.0, delay))


def _retry_after(response: httpx.Response) -> float | None:
    """Extract Telegram's ``retry_after`` hint from a 429 response."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return max(0.0, float(header))
        except (TypeError, ValueError):
            pass

    try:
        payload = response.json()
    except Exception:
        return None

    if isinstance(payload, dict):
        parameters = payload.get("parameters")
        if isinstance(parameters, dict):
            value = parameters.get("retry_after")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0.0, float(value))
    return None


def _response_ok(response: httpx.Response) -> bool:
    """Return ``True`` when a 200 response reports ``"ok": true``."""
    try:
        payload = response.json()
    except Exception:
        # A 200 with an unparseable body is unusual but not worth failing on.
        return True
    if isinstance(payload, dict) and "ok" in payload:
        return bool(payload["ok"])
    return True


def _body_preview(response: httpx.Response, limit: int = 300) -> str:
    """Return a short, log-safe preview of a response body."""
    try:
        return response.text[:limit]
    except Exception:  # pragma: no cover - defensive
        return "<unreadable>"


async def _sleep(delay: float) -> None:
    """Sleep for ``delay`` seconds, skipping zero/negative delays."""
    if delay > 0:
        await asyncio.sleep(delay)


async def send_lead_alert(
    lead_data: dict,
    original_text: str,
    source_chat: str,
    msg_link: str | None = None,
    *,
    config: TelegramConfig | None = None,
    client: httpx.AsyncClient | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    chat_id: str | None = None,
) -> bool:
    """Format and deliver a lead alert to Telegram.

    Transient failures (network errors, HTTP 429, HTTP 5xx) are retried with
    exponential backoff; a 429's ``retry_after`` hint is honoured when present.
    Non-retryable 4xx responses fail immediately.

    Args:
        lead_data: Classifier output passed to :func:`format_alert`.
        original_text: Raw message body.
        source_chat: Human-readable source chat name.
        msg_link: Optional direct link to the original message.
        config: Optional settings override. When omitted, ``.env`` is loaded
            asynchronously via :func:`load_telegram_config` and the destination
            chat is resolved via :func:`resolve_alert_chat_id` using the lead's
            category.
        client: Optional pre-built ``httpx.AsyncClient`` (used by tests). When
            omitted, a short-lived client is created and closed per call.
        max_retries: Additional attempts for transient failures. ``0`` means a
            single attempt.
        chat_id: Optional explicit destination chat id. When given it overrides
            both ``config.chat_id`` and category routing.

    Returns:
        ``True`` when Telegram accepted the message, ``False`` on any failure
        (including missing configuration). Never raises.
    """
    cfg = config if config is not None else await load_telegram_config()
    if not cfg.bot_token:
        LOGGER.warning(
            "Telegram alerting is not configured — set TELEGRAM_BOT_TOKEN and "
            f"{DEFAULT_CHAT_ENV} (or ALERT_CHAT_*) in .env to enable it."
        )
        return False

    destination = _resolve_destination(cfg, lead_data, chat_id, config)
    if not destination:
        return False

    text = format_alert(lead_data, original_text, source_chat, msg_link)
    payload: dict[str, Any] = {
        "chat_id": destination,
        "text": text,
        "parse_mode": PARSE_MODE,
        "disable_web_page_preview": True,
    }
    url = _endpoint(cfg.bot_token)

    attempts = max(0, int(max_retries)) + 1
    owns_client = client is None
    http: httpx.AsyncClient | None = None

    if client is not None:
        http = client
    else:
        proxy_url = resolve_proxy_url()
        if proxy_url is not None:
            LOGGER.info(
                "Routing Telegram alerts through proxy %s.",
                proxy_url.rsplit("@", 1)[-1],
            )
        try:
            http = httpx.AsyncClient(proxy=proxy_url, timeout=cfg.timeout)
        except Exception:
            # A missing SOCKS backend or an invalid proxy URL must not crash
            # the scraping pipeline.
            LOGGER.exception("Could not create the Telegram HTTP client.")
            return False

    try:
        for attempt in range(1, attempts + 1):
            try:
                response = await http.post(url, json=payload)
            except httpx.RequestError as exc:
                # Timeouts, DNS failures, connection resets — all transient.
                LOGGER.warning(
                    "Telegram transport error on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    await _sleep(_backoff(attempt))
                    continue
                LOGGER.error(
                    "Telegram alert failed after %d attempt(s).", attempts
                )
                return False
            except Exception:
                # Never let an unexpected transport bug kill the pipeline.
                LOGGER.exception(
                    "Unexpected error while sending a Telegram alert."
                )
                return False

            status = response.status_code

            if status == 200 and _response_ok(response):
                LOGGER.info(
                    "Telegram alert delivered to chat %s.", destination
                )
                return True

            if status == 429:
                delay = _retry_after(response)
                if delay is None:
                    delay = _backoff(attempt)
                delay = min(MAX_RETRY_DELAY, delay)
                LOGGER.warning(
                    "Telegram rate limit (429) on attempt %d/%d; "
                    "retrying in %.2fs.",
                    attempt,
                    attempts,
                    delay,
                )
                if attempt < attempts:
                    await _sleep(delay)
                    continue
                LOGGER.error(
                    "Telegram rate limit persisted after %d attempt(s).",
                    attempts,
                )
                return False

            if 500 <= status < 600:
                delay = _backoff(attempt)
                LOGGER.warning(
                    "Telegram server error %d on attempt %d/%d; "
                    "retrying in %.2fs.",
                    status,
                    attempt,
                    attempts,
                    delay,
                )
                if attempt < attempts:
                    await _sleep(delay)
                    continue
                LOGGER.error(
                    "Telegram server error persisted after %d attempt(s).",
                    attempts,
                )
                return False

            # Any other 4xx (bad token, bad chat id, malformed HTML, ...) is
            # permanent — retrying would only waste time.
            LOGGER.error(
                "Telegram rejected the alert (status=%d): %s",
                status,
                _body_preview(response),
            )
            return False
    finally:
        if owns_client and http is not None:
            await http.aclose()

    return False


def _resolve_destination(
    config: TelegramConfig,
    lead_data: object,
    explicit_chat_id: str | None,
    explicit_config: TelegramConfig | None,
) -> str:
    """Choose the destination chat id for one alert dispatch.

    Precedence: an explicit ``chat_id`` argument, then an explicitly supplied
    :class:`TelegramConfig`, then category-based environment routing.

    Returns:
        The chat id to target, or ``""`` when nothing usable is configured
        (the caller then skips the dispatch).
    """
    if explicit_chat_id is not None:
        value = explicit_chat_id.strip()
        if not value:
            LOGGER.warning(
                "An explicit chat_id was supplied but is blank — skipping alert."
            )
        return value

    if explicit_config is not None and config.chat_id:
        return config.chat_id

    data: Mapping[str, Any] = (
        lead_data if isinstance(lead_data, Mapping) else {}
    )
    raw_category = data.get("category")
    category = raw_category.strip() if isinstance(raw_category, str) else None

    try:
        return resolve_alert_chat_id(category)
    except RuntimeError as exc:
        LOGGER.warning("Telegram alert skipped: %s", exc)
        return ""


__all__ = [
    "API_BASE",
    "CATEGORY_CHAT_ENV",
    "DEFAULT_CATEGORY",
    "DEFAULT_CHAT_ENV",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_PROXY_TYPE",
    "DEFAULT_TIMEOUT",
    "DIRECT_PROXY_HOSTS",
    "ENV_FILE",
    "FALLBACK_CHAT_ENV",
    "MAX_CONFIDENCE",
    "MAX_MESSAGE_LENGTH",
    "MAX_RETRY_DELAY",
    "MAX_SNIPPET_CHARS",
    "MIN_CONFIDENCE",
    "PARSE_MODE",
    "PROXY_HOST_ENV",
    "PROXY_PASS_ENV",
    "PROXY_PORT_ENV",
    "PROXY_TYPE_ENV",
    "PROXY_USER_ENV",
    "RETRY_BASE_DELAY",
    "SUPPORTED_PROXY_TYPES",
    "TelegramConfig",
    "format_alert",
    "load_telegram_config",
    "read_telegram_config",
    "resolve_alert_chat_id",
    "resolve_proxy_url",
    "send_lead_alert",
]
