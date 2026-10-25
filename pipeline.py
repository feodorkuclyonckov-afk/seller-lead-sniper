"""Sprint 5 — master pipeline orchestration for the Lead Sniper.

Purpose
-------
Earlier sprints built four independent components:

* :mod:`filters` — cheap local pre-filter (length / stop words / triggers).
* :mod:`db` — async SQLite deduplication cache.
* :mod:`classifier` — async LLM semantic lead classifier.
* :mod:`notifier` — async Telegram alert dispatcher.

This module wires them together into a single, ordered, fail-safe pipeline so
that both the CLI runner (``main.py``) and the integration tests only ever call
one entry point: :func:`process_raw_message`.

Execution order
---------------
1. **Local filter** — reject obvious noise before any paid API call.
2. **Deduplication** — drop reposts already recorded in the local cache.
3. **LLM classifier** — decide whether the author is really asking for a
   contractor; drop non-leads.
4. **Alert dispatch** — send the qualified lead to the configured Telegram
   channel.

Every step emits a structured log line carrying the step number, outcome and the
message source, so a production run can be traced in the log file alone.

Status contract
---------------
:func:`process_raw_message` always returns a ``dict`` with a ``status`` key;
it never raises for runtime failures::

    {"status": "filtered", "reason": "keyword_or_length"}
    {"status": "duplicate"}
    {"status": "llm_unavailable"}
    {"status": "rejected_by_llm"}
    {"status": "dispatched", "lead_data": {...}, "delivered": bool}
    {"status": "error", "error": "..."}
"""

from __future__ import annotations

import logging
from typing import Any, Final

import classifier
import db as db_module
import notifier
from classifier import FALLBACK_CATEGORY
from db import DEFAULT_DB_PATH
from filters import passes_local_filter

LOGGER = logging.getLogger("lead_sniper.pipeline")

# --------------------------------------------------------------------------- #
# Status constants (imported by callers/tests to avoid magic strings)
# --------------------------------------------------------------------------- #

STATUS_FILTERED: Final[str] = "filtered"
STATUS_DUPLICATE: Final[str] = "duplicate"
STATUS_LLM_UNAVAILABLE: Final[str] = "llm_unavailable"
STATUS_REJECTED_BY_LLM: Final[str] = "rejected_by_llm"
STATUS_DISPATCHED: Final[str] = "dispatched"
STATUS_ERROR: Final[str] = "error"

#: Reason string reported when Step 1 rejects a message.
FILTER_REASON: Final[str] = "keyword_or_length"

#: Maximum number of characters of the raw message kept in log lines.
LOG_PREVIEW_CHARS: Final[int] = 90


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def _preview(text: Any, limit: int = LOG_PREVIEW_CHARS) -> str:
    """Return a single-line, length-capped preview of ``text`` for logs."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _source_label(source_chat: Any) -> str:
    """Return a log-safe source label (never empty)."""
    return _preview(source_chat or "unknown", 60)


# --------------------------------------------------------------------------- #
# Fallback lead construction
# --------------------------------------------------------------------------- #


def _fallback_lead(text: str) -> dict[str, Any]:
    """Build a low-confidence lead used when the LLM stage is unavailable.

    Only used when the caller explicitly opts in via ``llm_fallback=True``.
    The lead is deliberately conservative (confidence ``1``) so a human
    reviews it, while still surfacing the raw request that passed the local
    filter.
    """
    words = " ".join(text.split())[:60]
    return {
        "is_lead": True,
        "category": FALLBACK_CATEGORY,
        "summary": words or "Требует ручной проверки",
        "confidence": 1,
    }


# --------------------------------------------------------------------------- #
# Public orchestrator
# --------------------------------------------------------------------------- #


async def process_raw_message(
    text: str,
    source_chat: str,
    msg_link: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    *,
    llm_fallback: bool = False,
) -> dict[str, Any]:
    """Run one raw Telegram message through the full four-step pipeline.

    Args:
        text: Raw message body scraped from Telegram.
        source_chat: Human-readable name of the originating chat.
        msg_link: Optional direct link to the original message.
        db_path: Path to the deduplication SQLite cache.
        llm_fallback: When ``True`` and the LLM stage returns ``None`` (disabled
            endpoint, API outage, malformed response), dispatch a conservative
            fallback lead instead of dropping the message. Defaults to ``False``
            (drop), which keeps alert precision high.

    Returns:
        A status dict as documented in the module docstring. Never raises for
        runtime problems — unexpected exceptions are caught and reported as
        ``{"status": "error", "error": "..."}``.
    """
    source = _source_label(source_chat)
    LOGGER.info(
        "Pipeline start | source=%s | chars=%s | preview=%r",
        source,
        len(text) if isinstance(text, str) else "?",
        _preview(text),
    )

    try:
        # ------------------------------------------------------------------ #
        # Step 1 — local rule-based filter
        # ------------------------------------------------------------------ #
        if not passes_local_filter(text):
            LOGGER.info(
                "Step 1/4 local filter REJECTED | source=%s | reason=%s",
                source,
                FILTER_REASON,
            )
            return {"status": STATUS_FILTERED, "reason": FILTER_REASON}
        LOGGER.debug("Step 1/4 local filter passed | source=%s", source)

        # ------------------------------------------------------------------ #
        # Step 2 — deduplication (records the message when new)
        # ------------------------------------------------------------------ #
        if await db_module.is_duplicate(text, db_path):
            LOGGER.info(
                "Step 2/4 duplicate skipped | source=%s", source
            )
            return {"status": STATUS_DUPLICATE}
        LOGGER.debug("Step 2/4 unique message recorded | source=%s", source)

        # ------------------------------------------------------------------ #
        # Step 3 — LLM semantic classification
        # ------------------------------------------------------------------ #
        lead_data = await classifier.classify_message(text)

        if lead_data is None:
            if llm_fallback:
                lead_data = _fallback_lead(text)
                LOGGER.warning(
                    "Step 3/4 LLM unavailable — dispatching fallback lead "
                    "| source=%s",
                    source,
                )
            else:
                LOGGER.info(
                    "Step 3/4 LLM unavailable/disabled — message dropped "
                    "| source=%s",
                    source,
                )
                return {"status": STATUS_LLM_UNAVAILABLE}

        elif not lead_data.get("is_lead"):
            LOGGER.info(
                "Step 3/4 rejected by LLM | source=%s | category=%s "
                "| confidence=%s",
                source,
                lead_data.get("category"),
                lead_data.get("confidence"),
            )
            return {"status": STATUS_REJECTED_BY_LLM}

        LOGGER.debug(
            "Step 3/4 classified as lead | source=%s | category=%s "
            "| confidence=%s",
            source,
            lead_data.get("category"),
            lead_data.get("confidence"),
        )

        # ------------------------------------------------------------------ #
        # Step 4 — alert dispatch
        # ------------------------------------------------------------------ #
        delivered = await notifier.send_lead_alert(
            lead_data, text, source_chat, msg_link
        )
        if delivered:
            LOGGER.info(
                "Step 4/4 alert DISPATCHED | source=%s | category=%s "
                "| confidence=%s",
                source,
                lead_data.get("category"),
                lead_data.get("confidence"),
            )
        else:
            LOGGER.warning(
                "Step 4/4 dispatch FAILED (notifier returned False) "
                "| source=%s | category=%s",
                source,
                lead_data.get("category"),
            )

        return {
            "status": STATUS_DISPATCHED,
            "lead_data": lead_data,
            "delivered": bool(delivered),
        }

    except Exception as exc:  # noqa: BLE001 - pipeline must never crash a scraper
        LOGGER.exception(
            "Pipeline ERROR | source=%s | error=%s", source, exc
        )
        return {"status": STATUS_ERROR, "error": str(exc)}


__all__ = [
    "FILTER_REASON",
    "LOG_PREVIEW_CHARS",
    "STATUS_DISPATCHED",
    "STATUS_DUPLICATE",
    "STATUS_ERROR",
    "STATUS_FILTERED",
    "STATUS_LLM_UNAVAILABLE",
    "STATUS_REJECTED_BY_LLM",
    "process_raw_message",
]
