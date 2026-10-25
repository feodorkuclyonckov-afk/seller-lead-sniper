"""Sprint 2 — local, rule-based pre-filter for scraped Telegram messages.

Purpose
-------
Only a small fraction of scraped messages are actual clients looking for a
contractor. Running every message through a paid LLM is wasteful, so this
module applies three cheap checks *before* the message ever leaves the
machine:

1. **Length** — trivially short ("+", "цена?") and wall-of-text posts are
   never useful leads.
2. **Stop words** — seller self-promotion, course ads, "продам базу" spam.
3. **Trigger words** — explicit Russian demand/inquiry stems ("ищу",
   "посоветуйте", "кто возит", ...). No trigger means no lead.

All matching is performed on ``text.lower()`` and is substring-based (stems
are used on purpose so that "ищу"/"ищет"/"ищем" all match with one token).

Public API
----------
``passes_local_filter(text)``
    ``True`` when the message is worth sending to the LLM classifier.
"""

from __future__ import annotations

from typing import Final

#: Minimum meaningful message length (characters) — shorter posts are noise.
MIN_LENGTH: Final[int] = 25

#: Maximum message length (characters) — longer posts are usually essays or
#: channel digests rather than concrete requests.
MAX_LENGTH: Final[int] = 850

#: Stems that signal an explicit demand, inquiry or request for a contractor.
#: Stored lowercase; each entry is matched as a plain substring.
TRIGGER_WORDS: Final[tuple[str, ...]] = (
    # Direct search / request verbs
    "ищу",
    "ищем",
    "ищет",
    "посоветуйте",
    "порекомендуйте",
    "подскажите",
    "порекомендуете",
    "посоветуете",
    "кто подскажет",
    "кто посоветует",
    "кто порекомендует",
    # "Who does X?" phrasing
    "кто занимается",
    "кто возит",
    "кто может",
    "кто поможет",
    "кто делает",
    "кто работает с",
    "кто оказывает",
    "кто-нибудь",
    "есть кто",
    # Explicit need
    "нужен",
    "нужна",
    "нужно",
    "нужны",
    "требуется",
    "требуются",
    "интересует",
    "интересуют",
    "рассматриваю",
    # Sourcing vocabulary typical for cargo / e-commerce / marketing
    "ищу подрядчика",
    "ищем подрядчика",
    "ищу поставщика",
    "ищем поставщика",
    "ищу перевозчика",
    "ищу карго",
    "нужен карго",
    "нужна доставка",
    "нужна перевозка",
    "ищу маркетолога",
    "ищу таргетолога",
    "нужен специалист",
    "нужны услуги",
    "нужна помощь",
    "где заказать",
    "где найти",
    "где искать",
)

#: Stems that signal spam or seller self-promotion. Anything matching one of
#: these is discarded before the trigger check, so an ad that also contains
#: "нужен" cannot slip through.
STOP_WORDS: Final[tuple[str, ...]] = (
    # Self-promotion / "I offer my services"
    "предлагаю услуги",
    "предлагаю сотрудничество",
    "мои услуги",
    "моя команда",
    "мы предлагаем",
    "оказываем услуги",
    "оказываю услуги",
    "готовы предложить",
    "готов предложить",
    "возьму на ведение",
    "возьму ваш проект",
    # Portfolio / case-study bait
    "мои кейсы",
    "мой кейс",
    "кейсы в закрепе",
    "портфолио",
    "отзывы в профиле",
    # Sales funnel / CTA spam
    "бесплатная консультация",
    "бесплатный разбор",
    "пишите в личку",
    "писать в личку",
    "ссылка в профиле",
    "приглашаю в канал",
    "приглашаю в чат",
    "подписывайтесь",
    "записывайтесь",
    "мест ограничено",
    "только сегодня",
    "скидка",
    "акция",
    "вебинар",
    "интенсив",
    "марафон",
    # Course / mentorship sales
    "набираю учеников",
    "набираю команду",
    "обучение с нуля",
    "наставничество",
    "курс по",
    "продаю курс",
    "моя школа",
    "личное сопровождение",
    # Lead-database / growth-hack spam
    "продам базу",
    "продаю базу",
    "свежие базы",
    "база клиентов",
    "база контактов",
    "базы telegram",
    "партнерская программа",
    "партнёрская программа",
    "реферальная",
    "накрутка",
    "накручу",
    "гарантированный результат",
    "ищу клиентов",
    "заработай",
    "доход от",
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
    3. Trigger-word check — reject if *none* of :data:`TRIGGER_WORDS` occur in
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

    # 2. Seller spam / self-promotion gate.
    if _first_match(lowered, STOP_WORDS) is not None:
        return False

    # 3. Explicit demand gate.
    if _first_match(lowered, TRIGGER_WORDS) is None:
        return False

    return True
