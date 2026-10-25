"""Sprint 3 — async LLM lead classifier for scraped Telegram messages.

Purpose
-------
Sprint 2's :mod:`filters` module removes obvious noise locally, but the
remaining messages still need a semantic judgement: is the author *asking for
a contractor* (a lead) or merely chatting, advertising, or offering their own
services? That judgement is delegated to an OpenAI-compatible chat model
(DeepSeek by default).

The classifier is deliberately defensive. A scraping pipeline must never die
because a third-party API timed out, rate-limited us, or returned malformed
JSON, so every failure path returns ``None`` instead of raising.

Public API
----------
``LLMConfig``
    Immutable settings container (``api_key``, ``base_url``, ``model``).
``load_llm_config(env_file)``
    Asynchronously load ``.env`` and return an :class:`LLMConfig` with clean
    fallbacks. Never raises.
``read_llm_config()``
    Synchronous variant that reads the current process environment.
``classify_message(text, ...)``
    Classify one message. Returns a normalised ``dict`` on success or ``None``
    on any failure.
``aclose()``
    Close the cached ``AsyncOpenAI`` client (call on shutdown).

Output schema
-------------
``{"is_lead": bool, "category": str, "summary": str, "confidence": int}``

``category`` is always one of :data:`CATEGORIES`; ``summary`` is a 5–8 word
Russian phrase; ``confidence`` is clamped to ``1..10``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)

LOGGER = logging.getLogger("lead_sniper.classifier")

# --------------------------------------------------------------------------- #
# Configuration defaults
# --------------------------------------------------------------------------- #

#: Default OpenAI-compatible endpoint (DeepSeek).
DEFAULT_BASE_URL: Final[str] = "https://api.deepseek.com"

#: Default chat model served by the endpoint above.
DEFAULT_MODEL: Final[str] = "deepseek-chat"

#: Per-request timeout in seconds. Kept short so a stalled API cannot block the
#: scraping loop for long.
DEFAULT_TIMEOUT: Final[float] = 30.0

#: Number of *additional* attempts for transient failures (timeouts, connection
#: resets, HTTP 429).
DEFAULT_MAX_RETRIES: Final[int] = 2

#: Base delay for exponential backoff between retries, in seconds.
RETRY_BASE_DELAY: Final[float] = 0.75

#: Hard cap on the number of words kept in ``summary``.
MAX_SUMMARY_WORDS: Final[int] = 8

#: Inclusive bounds for the ``confidence`` field.
MIN_CONFIDENCE: Final[int] = 1
MAX_CONFIDENCE: Final[int] = 10

#: Allowed categories, in the exact spelling expected downstream.
CATEGORIES: Final[tuple[str, ...]] = (
    "Фулфилмент",
    "Карго / Логистика",
    "Маркетинг / Трафик",
    "Дизайн / Контент",
    "Другое",
)

#: Category used when the model invents an unknown label.
FALLBACK_CATEGORY: Final[str] = "Другое"

#: Location of the project ``.env`` file (next to this module).
ENV_FILE: Final[Path] = Path(__file__).resolve().with_name(".env")

#: System prompt describing the qualification task and the JSON contract.
SYSTEM_PROMPT: Final[str] = """\
Ты — опытный B2B-аналитик агентства, которое оказывает услуги фулфилмента,
карго-логистики, маркетинга и дизайна для e-commerce.

Твоя задача — по сообщению из Telegram-чата определить, является ли автор
потенциальным клиентом (заказчиком), который ищет подрядчика или исполнителя.

Категории (используй ровно одну из них):
- "Фулфилмент" — хранение, сборка заказов, упаковка, обработка возвратов, склад.
- "Карго / Логистика" — доставка из Китая и других стран, карго, перевозки, таможня.
- "Маркетинг / Трафик" — таргет, реклама, SEO, продвижение, трафик, аналитика.
- "Дизайн / Контент" — дизайн сайта и креативов, фото, видео, тексты, контент.
- "Другое" — запрос не относится к перечисленным услугам.

Правила:
1. is_lead = true только если автор ищет исполнителя, подрядчика или поставщика услуг.
2. is_lead = false для самопрезентаций, рекламы курсов, продажи баз, вакансий,
   флуда, вопросов от самих исполнителей и сообщений без запроса.
3. summary — краткое описание запроса на русском, строго 5–8 слов, без воды.
4. confidence — целое число от 1 до 10: насколько уверенно сообщение является лидом.
5. Отвечай ТОЛЬКО валидным JSON-объектом без markdown и пояснений, например:
{"is_lead": true, "category": "Карго / Логистика", "summary": "Поиск карго из Китая в Москву", "confidence": 9}
"""

#: Case-insensitive lookup used to canonicalise model-provided categories.
_CATEGORY_LOOKUP: Final[dict[str, str]] = {
    category.lower(): category for category in CATEGORIES
}

#: Matches a leading ```json / ``` fence and a trailing ``` fence.
_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"^```(?:json)?\s*|\s*```$", re.IGNORECASE
)

#: Truthy / falsy spellings accepted for ``is_lead``.
_TRUE_WORDS: Final[frozenset[str]] = frozenset({"true", "1", "yes", "y", "да"})
_FALSE_WORDS: Final[frozenset[str]] = frozenset({"false", "0", "no", "n", "нет"})


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Immutable LLM settings with safe defaults.

    Attributes:
        api_key: Bearer token for the endpoint. Empty means "not configured".
        base_url: OpenAI-compatible base URL.
        model: Chat model name.
        timeout: Per-request timeout in seconds.
    """

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """``True`` when an API key is present, i.e. classification can run."""
        return bool(self.api_key)


def _read(name: str, default: str = "") -> str:
    """Return a stripped environment value, falling back to ``default``."""
    value = (os.getenv(name) or "").strip()
    return value or default


def read_llm_config() -> LLMConfig:
    """Build an :class:`LLMConfig` from the current process environment.

    Missing or blank variables fall back to :data:`DEFAULT_BASE_URL` and
    :data:`DEFAULT_MODEL`; a missing key simply leaves the classifier disabled.
    This function never raises.
    """
    return LLMConfig(
        api_key=_read("LLM_API_KEY"),
        base_url=_read("LLM_BASE_URL", DEFAULT_BASE_URL),
        model=_read("LLM_MODEL", DEFAULT_MODEL),
    )


async def load_llm_config(env_file: str | Path | None = None) -> LLMConfig:
    """Load ``.env`` off the event loop and return the resulting config.

    ``python-dotenv`` performs blocking file I/O, so it is dispatched to a
    worker thread. Existing process variables win (``override=False``), which
    keeps container/CI overrides working. Any error while reading the file is
    logged and ignored — the caller still receives usable defaults.

    Args:
        env_file: Optional path to the dotenv file. Defaults to the project
            ``.env`` next to this module.

    Returns:
        An :class:`LLMConfig`; never raises.
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

    return read_llm_config()


# --------------------------------------------------------------------------- #
# Client lifecycle
# --------------------------------------------------------------------------- #

_client: AsyncOpenAI | None = None
_client_signature: tuple[str, str, float] | None = None


def _get_client(config: LLMConfig) -> AsyncOpenAI:
    """Return a cached ``AsyncOpenAI`` client for ``config``.

    The client is recreated only when the key, base URL or timeout changes.
    ``max_retries=0`` is passed to the SDK because retries are handled here,
    which keeps the backoff policy explicit and testable.
    """
    global _client, _client_signature

    signature = (config.api_key, config.base_url, config.timeout)
    if _client is None or _client_signature != signature:
        _client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )
        _client_signature = signature
        LOGGER.debug(
            "Created AsyncOpenAI client (base_url=%s, model=%s).",
            config.base_url,
            config.model,
        )
    return _client


async def aclose() -> None:
    """Close the cached client and release its HTTP connections."""
    global _client, _client_signature

    client, _client = _client, None
    _client_signature = None
    if client is None:
        return
    try:
        await client.close()
    except Exception:  # pragma: no cover - shutdown best effort
        LOGGER.debug("Error while closing the LLM client.", exc_info=True)


# --------------------------------------------------------------------------- #
# Response parsing / normalisation
# --------------------------------------------------------------------------- #


def _extract_content(response: Any) -> str | None:
    """Return the first choice's text content, or ``None`` when absent."""
    try:
        choices = response.choices
        if not choices:
            return None
        content = choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None

    if not isinstance(content, str):
        return None
    content = content.strip()
    return content or None


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Parse ``raw`` into a JSON object, tolerating fences and stray prose.

    Models occasionally wrap JSON in ```json fences or add a sentence before
    it even when ``response_format`` is set. Both cases are handled: fences are
    stripped first, then a fallback scan extracts the outermost ``{...}``.
    """
    text = _FENCE_RE.sub("", raw.strip()).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return payload if isinstance(payload, dict) else None


def _coerce_bool(value: Any) -> bool | None:
    """Coerce common truthy/falsy spellings to ``bool``; ``None`` if invalid."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
    return None


def _coerce_confidence(value: Any) -> int | None:
    """Coerce ``value`` to an int clamped to ``1..10``; ``None`` if invalid."""
    if isinstance(value, bool):
        return None
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, number))


def _normalise_summary(value: Any) -> str | None:
    """Collapse whitespace and cap the summary at :data:`MAX_SUMMARY_WORDS`."""
    if not isinstance(value, str):
        return None
    words = value.split()
    if not words:
        return None
    return " ".join(words[:MAX_SUMMARY_WORDS])


def _normalise_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate and normalise a decoded JSON payload.

    Returns the canonical four-key dict, or ``None`` when a required field is
    missing or cannot be coerced. Unknown categories fall back to
    :data:`FALLBACK_CATEGORY`; out-of-range confidence values are clamped.
    """
    is_lead = _coerce_bool(payload.get("is_lead"))
    if is_lead is None:
        return None

    raw_category = payload.get("category")
    category = FALLBACK_CATEGORY
    if isinstance(raw_category, str):
        category = _CATEGORY_LOOKUP.get(
            raw_category.strip().lower(), FALLBACK_CATEGORY
        )

    summary = _normalise_summary(payload.get("summary"))
    if summary is None:
        return None

    confidence = _coerce_confidence(payload.get("confidence"))
    if confidence is None:
        return None

    return {
        "is_lead": is_lead,
        "category": category,
        "summary": summary,
        "confidence": confidence,
    }


# --------------------------------------------------------------------------- #
# Public classification entry point
# --------------------------------------------------------------------------- #


async def classify_message(
    text: str,
    *,
    client: Any | None = None,
    config: LLMConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any] | None:
    """Classify one Telegram message as a potential B2B lead.

    Args:
        text: Raw message body.
        client: Optional pre-built client (used by tests). Must expose
            ``chat.completions.create``. When omitted, a cached
            :class:`openai.AsyncOpenAI` client is used.
        config: Optional settings override. When omitted, ``.env`` is loaded
            asynchronously via :func:`load_llm_config`.
        max_retries: Additional attempts for transient failures (timeouts,
            connection errors, HTTP 429). ``0`` means a single attempt.

    Returns:
        ``{"is_lead": bool, "category": str, "summary": str,
        "confidence": int}`` on success, or ``None`` when the message is empty,
        the API key is missing, the request fails, or the response cannot be
        parsed/validated.

    Raises:
        TypeError: if ``text`` is not a ``str`` (programmer error, not an API
            failure).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"classify_message() expects str, got {type(text).__name__}"
        )

    message = text.strip()
    if not message:
        LOGGER.debug("Empty message — nothing to classify.")
        return None

    cfg = config if config is not None else await load_llm_config()
    if not cfg.is_configured:
        LOGGER.warning(
            "LLM_API_KEY is not set — skipping classification. "
            "Add it to .env to enable the LLM stage."
        )
        return None

    api = client if client is not None else _get_client(cfg)
    attempts = max(0, int(max_retries)) + 1

    for attempt in range(1, attempts + 1):
        try:
            response = await api.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Сообщение из Telegram-чата:\n\n{message}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            # Transient: retry with exponential backoff while attempts remain.
            LOGGER.warning(
                "Transient LLM error on attempt %d/%d: %s",
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if delay > 0:
                    await asyncio.sleep(delay)
                continue
            LOGGER.error("LLM request failed after %d attempt(s).", attempts)
            return None
        except APIStatusError as exc:
            # Non-retryable HTTP error (bad request, auth, server error, ...).
            LOGGER.error(
                "LLM API rejected the request (status=%s): %s",
                getattr(exc, "status_code", "?"),
                exc,
            )
            return None
        except OpenAIError as exc:
            LOGGER.error("OpenAI client error: %s", exc)
            return None
        except Exception:
            # Never let an unexpected SDK/transport bug kill the pipeline.
            LOGGER.exception("Unexpected error while calling the LLM API.")
            return None

        content = _extract_content(response)
        if content is None:
            LOGGER.warning("LLM returned an empty response.")
            return None

        payload = _extract_json_object(content)
        if payload is None:
            LOGGER.warning("LLM returned invalid JSON: %r", content[:200])
            return None

        result = _normalise_payload(payload)
        if result is None:
            LOGGER.warning(
                "LLM JSON did not match the expected schema: %r", payload
            )
            return None

        LOGGER.debug(
            "Classified message as lead=%s category=%s confidence=%d.",
            result["is_lead"],
            result["category"],
            result["confidence"],
        )
        return result

    return None


__all__ = [
    "CATEGORIES",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "ENV_FILE",
    "FALLBACK_CATEGORY",
    "LLMConfig",
    "MAX_CONFIDENCE",
    "MAX_SUMMARY_WORDS",
    "MIN_CONFIDENCE",
    "SYSTEM_PROMPT",
    "aclose",
    "classify_message",
    "load_llm_config",
    "read_llm_config",
]
