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

Calibration notes (live-chat audit)
-----------------------------------
The trigger list was broadened with the conversational vocabulary real sellers
use in WB/Ozon chats: "минус", "в ноль", "маржинальност", "слив", "съедает",
"кассовый разрыв", "скинул цену", "уронил цену", "акци", "комисси",
"перерасчет", "квв", "платная приемка", "покатушк", "обезличк",
"потеряли товар", "вб списал", "озон списал", "выплат", "задерж".

Because those stems also occur in service ads (fulfillment, analytics, cargo,
HiSeller-style SaaS), the stop-word list was extended to reject *obvious
commercial spam* before the trigger check: agency offers, card design,
recruitment ads, bots, crypto/gift-card shops and reviews exchange. A genuine
seller complaint that merely mentions "логистик" or "штраф" is not blocked.

``MIN_LENGTH`` was lowered from 30 to 20 so concise two-sentence complaints
such as "Опять удержали за логистику х3" are not dropped by the length gate.

Public API
----------
``passes_local_filter(text)``
    ``True`` when the message is worth sending to the LLM classifier.
"""

from __future__ import annotations

from typing import Final

#: Minimum meaningful message length (characters) — shorter posts are noise.
#: Lowered from 30 to 20 so concise seller complaints survive the length gate.
MIN_LENGTH: Final[int] = 20

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
    "скинул цен",
    "уронил",
    "уронил цен",
    "демпинг",
    "акци",
    # Logistics & tariffs
    "логистик",
    "коэффициент",
    "кхв",
    "квв",
    "коэф",
    "хранен",
    "приемк",
    "платная приемк",
    "завысил габарит",
    "перемер",
    "покатушк",
    "обезличк",
    "потеряли товар",
    "списал",
    "вб списал",
    "озон списал",
    # Penalties & deductions
    "штраф",
    "удержан",
    "кбжу",
    "самовыкуп",
    "отчет реализаци",
    "минус в отчете",
    "комисси",
    "перерасчет",
    "перерасчёт",
    # Unit-economics drain
    "в минус",
    "минус",
    "убыт",
    "убыток",
    "в ноль",
    "марж",
    "маржинальност",
    "слив",
    "съедает",
    "съел прибыль",
    "себестоимост",
    "кассов",
    "кассовый разрыв",
    # Payouts & delays (observed in live seller chats)
    "выплат",
    "задерж",
)

#: Stems that signal marketplace spam or agency offers. Anything matching one
#: of these is discarded before the trigger check, so an agency ad that also
#: mentions "штраф" or "логистик" cannot slip through.
#:
#: The list is deliberately limited to *obvious commercial spam*: service
#: offers, card design, recruitment ads, bots, crypto/gift-card shops and
#: reviews exchange. It must never block a genuine seller complaint.
STOP_WORDS: Final[tuple[str, ...]] = (
    # Agency / service offers
    "предлагаю услуги",
    "выведу в топ",
    "накрутк",
    "инфографик",
    "оформление карточек",
    "курс по",
    "пишите в лс",
    "пишите в личку",
    "пишите в личные",
    "в личные сообщения",
    "в лс",
    "обращайтесь",
    "контакт:",
    "контакты:",
    "телеграм:",
    "telegram:",
    "whatsapp",
    "наш сайт",
    "сайт:",
    "@gmail",
    "gmail.com",
    "наш опыт",
    "сопровождаем",
    "команды, которая",
    "бесплатный сервис",
    "синхронизац",
    "подписк",
    "официальный партнёр",
    "официальный партнер",
    "hiseller",
    "сэкономить десятки",
    "для расчета стоимости",
    "рассчитать стоимость",
    "задать вопрос",
    "больше отзывов",
    "отзывы о нас",
    "актуальный пост",
    "что изменилось",
    "собрали основные изменения",
    "продвижение карточек",
    "продвижению карточек",
    "услуги по продвижению",
    "#продвижение",
    "#менеджер",
    "#дизайнер",
    "#фулфилмент",
    "#маркетплейсы",
    "#упаковка",
    "наш фулфилмент",
    "что мы делаем",
    "полный цикл",
    "ответственное хранение",
    "работа по фбо",
    "доверьтесь",
    "гарантируем",
    "работаем строго",
    "пишите менеджеру",
    "напишите нам",
    "напишите менеджеру",
    "связаться с менеджером",
    # Recruitment / job ads
    "предлагаю вакансию",
    "требуется ассистентка",
    "ищу помощницу",
    "ищу ответственного",
    "открыта вакансия",
    "подработка",
    "зарплат",
    "на руки",
    # Card design / SEO services
    "создам карточк",
    "создаём карточки",
    "создание карточек",
    "дизайн карточек",
    "карточки или инфографика",
    "ищу дизайнера",
    "seo",
    "ключевых слов",
    "оптимизац",
    "анализ продаж",
    "анализ прибыли",
    "точки роста",
    # Bots / crypto / gift cards / reviews exchange
    "куплю аккаунт",
    "обмен отзыв",
    "накрутка отзыв",
    "выкуп отзыв",
    "crypto",
    "крипт",
    "usdt",
    "giftcards",
    "prepaid",
    "биржа",
    "рассылка",
    "раскрутка",
    "подписчиков",
    "@bot",
    "_bot",
    "чат-бот",
    "бот получил",
    "бот создан",
    "розыгрыш",
    "в подарок",
    # Wholesale / cargo / currency ads
    "оптовые цены",
    "полный прайс",
    "отправим полный прайс",
    "без посредников",
    "мы производитель",
    "можно заказать",
    "курс юаня",
    "обмен юаней",
    "перевод денег",
    "гарантия безопасности",
    "байер из китая",
    "байер китай",
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
