"""Sprint 2 — local, rule-based pre-filter for scraped Telegram messages.

Purpose
-------
Only a small fraction of scraped messages are marketplace sellers in real
pain: margin losses, unexpected fees and forced auto-actions. Running every
message through a paid LLM is wasteful, so this module applies three cheap
checks *before* the message ever leaves the machine:

1. **Length** — trivially short ("Где склад?") and wall-of-text posts are
   never useful leads.
2. **Stop words** — agency offers, growth-hack ads, course sales and
   "пишите в личку" spam.
3. **Trigger words** — stems describing WB/Ozon seller pain ("автоакци",
   "штраф", "в минус", ...). No trigger means no lead.

All matching is performed on ``text.lower()`` and is substring-based (stems
are used on purpose so that "штраф"/"штрафы"/"штрафа" all match one token).

Public API
----------
``passes_local_filter(text)``
    ``True`` when the message is worth sending to the LLM classifier.
"""

from __future__ import annotations

from typing import Final

#: Minimum meaningful message length (characters) — shorter posts are noise.
MIN_LENGTH: Final[int] = 30

#: Maximum message length (characters) — longer posts are usually essays or
#: channel digests rather than concrete complaints.
MAX_LENGTH: Final[int] = 900

#: Stems that signal an explicit WB/Ozon seller pain point: forced
#: auto-actions, tariff hikes, penalties and margin bleed.
#: Stored lowercase; each entry is matched as a plain substring.
TRIGGER_WORDS: Final[tuple[str, ...]] = (
    # Auto-actions & pricing pressure
    "автоакци",
    "принудительн",
    "скидк",
    "спп",
    "слетел цен",
    "сбросил цен",
    "демпинг",
    # Logistics & tariffs
    "логистик",
    "коэффициент",
    "кхв",
    "коэф",
    "хранен",
    "приемк",
    "завысил габарит",
    "перемер",
    # Penalties & deductions
    "штраф",
    "удержан",
    "кбжу",
    "самовыкуп",
    "отчет реализаци",
    "минус в отчете",
    # Unit-economics drain
    "в минус",
    "убыт",
    "слив марж",
    "себестоимост",
    "съел прибыль",
    "кассовый разрыв",
)

#: Stems that signal marketplace spam or agency offers. Anything matching one
#: of these is discarded before the trigger check, so an agency ad that also
#: mentions "штраф" or "логистик" cannot slip through.
STOP_WORDS: Final[tuple[str, ...]] = (
    "предлагаю услуги",
    "выведу в топ",
    "накрутк",
    "продвижени",
    "инфографик",
    "оформление карточек",
    "курс по",
    "байер",
    "фулфилмент москва",
    "пишите в лс",
    "пишите в личку",
)


def _first_match(text_lower: str, words: tuple[str, ...]) -> str | None:
    """Return the first keyword contained in ``text_lower``, else ``None``."""
    for word in words:
        if word in text_lower:
            return word
    return None


def passes_local_filter(text: str) -> bool:
    """Decide whether ``text`` is worth sending to the LLM classifier.

    Evaluation order (cheapest first):

    1. Length check — reject if ``len(text) < MIN_LENGTH`` or
       ``len(text) > MAX_LENGTH``.
    2. Stop-word check — reject if any :data:`STOP_WORDS` entry occurs in
       ``text.lower()``.
    3. Pain-point check — reject if *none* of :data:`TRIGGER_WORDS` occur in
       ``text.lower()``.

    Args:
        text: The raw message body.

    Returns:
        ``True`` only when the message survives all three checks.

    Raises:
        TypeError: if ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"passes_local_filter() expects str, got {type(text).__name__}"
        )

    # 1. Length gate.
    if len(text) < MIN_LENGTH or len(text) > MAX_LENGTH:
        return False

    lowered = text.lower()

    # 2. Marketplace spam / agency-offer gate.
    if _first_match(lowered, STOP_WORDS) is not None:
        return False

    # 3. Seller pain-point gate.
    if _first_match(lowered, TRIGGER_WORDS) is None:
        return False

    return True
