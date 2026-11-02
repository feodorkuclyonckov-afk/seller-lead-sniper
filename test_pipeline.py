"""Sprint 5 — end-to-end integration test for the master pipeline.

Run with::

    .venv\\Scripts\\python.exe test_pipeline.py

The suite exercises :func:`pipeline.process_raw_message` exactly as production
does — real local filter (Step 1) and real SQLite deduplication (Step 2) — while
the *external* services are mocked so no network access, API key or bot token is
required:

* **LLM classifier** — replaced by a deterministic coroutine that returns canned
  payloads per message, ``is_lead`` of ``False`` for ambiguous texts, and
  ``None`` for the simulated API failure.
* **Telegram notifier** — replaced by a recorder that asserts dispatch payloads.

A batch of 10 marketplace seller messages covers every terminal status: local
filter (spam and short noise), duplicate, LLM rejection, LLM unavailability,
and successful dispatch of real seller pain points. The suite then verifies the
counter distribution, dispatch gating and the defensive error path. Exit code
``0`` means every check passed; ``1`` means otherwise.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Final
from unittest import mock

import classifier
import db as db_module
import notifier
import pipeline

# --------------------------------------------------------------------------- #
# Fixtures — message bodies and their expected classification
# --------------------------------------------------------------------------- #

#: Forced auto-action and price dump below cost — a real lead.
LEAD_AUTO: Final[str] = (
    "Включили принудительную автоакцию на весь ассортимент, цены срезали "
    "ниже себестоимости, скидки рисуют сами, маржа уже в минус. "
    "Кто сталкивался, как отключить?"
)

#: Warehouse coefficient spike / logistics re-calculation — a real lead.
LEAD_LOGISTICS: Final[str] = (
    "С 11 числа WB поднял коэффициент склада до х5, логистика и хранение "
    "пересчитали задним числом, доплата выросла в три раза. "
    "Кто оспаривал такие начисления?"
)

#: Packaging dimension fine / deduction in the sales report — a real lead.
LEAD_FINE: Final[str] = (
    "Прилетел штраф за перемер габаритов: удержали 40 тысяч из отчёта "
    "реализации, хотя упаковка не менялась. Как доказать, что замер неверный?"
)

#: Agency ad / reviews boosting spam — must be rejected by the local filter.
SPAM_AGENCY: Final[str] = (
    "Предлагаю услуги по накрутке отзывов и продвижению карточек WB, "
    "пишите в личку, выведу в топ за месяц."
)

#: Casual chatter with no pain trigger — must be rejected by the local filter.
OFF_TOPIC: Final[str] = (
    "Сегодня отличная погода в Москве, все пошли гулять в парк"
)

#: Low-length query — must be rejected by the length gate.
SHORT_QUERY: Final[str] = "цена?"

AMBIGUOUS: Final[str] = (
    "Читаю новости про новые штрафы маркетплейсов, пока просто собираю "
    "информацию, сам пока не попал."
)

LLM_FAILURE: Final[str] = (
    "Логистика пересчитала хранение задним числом, доплата за приёмку "
    "выросла вдвое, не понимаю, откуда такой коэффициент."
)

#: Text used by the defensive error-path check (passes the local filter).
ERROR_TEXT: Final[str] = (
    "Автоакция срезала цену ниже себестоимости, продажи в минус, "
    "не пойму, как это отключить."
)

#: Text used by the fallback check (passes the local filter, LLM returns None).
FALLBACK_TEXT: Final[str] = (
    "Штраф за габариты удержали из отчёта реализации, логистика списала "
    "лишнее, как оспорить удержание?"
)

#: Deterministic classifier responses keyed by stripped message body.
CLASSIFIER_RESPONSES: Final[dict[str, dict[str, Any] | None]] = {
    LEAD_AUTO: {
        "is_lead": True,
        "category": "Автоакция / Цены",
        "summary": "Принудительная автоакция уронила цену ниже себестоимости",
        "confidence": 9,
    },
    LEAD_LOGISTICS: {
        "is_lead": True,
        "category": "Логистика / Хранение",
        "summary": "Скачок коэффициента склада и пересчёт логистики",
        "confidence": 8,
    },
    LEAD_FINE: {
        "is_lead": True,
        "category": "Штрафы / Удержания",
        "summary": "Штраф за перемер габаритов удержан из отчёта",
        "confidence": 9,
    },
    AMBIGUOUS: {
        "is_lead": False,
        "category": "Другое",
        "summary": "Автор только читает новости о штрафах",
        "confidence": 3,
    },
    LLM_FAILURE: None,
}

#: ``(label, text, source_chat, msg_link, expected_status, expected_reason)``.
BATCH: Final[tuple[tuple[str, str, str, str | None, str, str | None], ...]] = (
    (
        "tiny-noise",
        "+",
        "Барахолка",
        None,
        pipeline.STATUS_FILTERED,
        pipeline.FILTER_REASON,
    ),
    (
        "seller-spam",
        SPAM_AGENCY,
        "Маркетплейс Услуги",
        None,
        pipeline.STATUS_FILTERED,
        pipeline.FILTER_REASON,
    ),
    (
        "off-topic",
        OFF_TOPIC,
        "Болталка",
        None,
        pipeline.STATUS_FILTERED,
        pipeline.FILTER_REASON,
    ),
    (
        "valid-auto",
        LEAD_AUTO,
        "WB Селлеры Чат",
        "https://t.me/wb_sellers_chat/12345",
        pipeline.STATUS_DISPATCHED,
        None,
    ),
    (
        "repost-auto",
        LEAD_AUTO,
        "Ozon Селлеры Чат",
        None,
        pipeline.STATUS_DUPLICATE,
        None,
    ),
    (
        "ambiguous",
        AMBIGUOUS,
        "Маркетплейс Новости",
        "https://t.me/marketplace_news/222",
        pipeline.STATUS_REJECTED_BY_LLM,
        None,
    ),
    (
        "valid-logistics",
        LEAD_LOGISTICS,
        "WB Логистика Чат",
        "https://t.me/wb_logistics_chat/777",
        pipeline.STATUS_DISPATCHED,
        None,
    ),
    (
        "llm-down",
        LLM_FAILURE,
        "Ozon Хранение Чат",
        None,
        pipeline.STATUS_LLM_UNAVAILABLE,
        None,
    ),
    (
        "valid-fine",
        LEAD_FINE,
        "WB Штрафы Чат",
        "https://t.me/wb_fines_chat/4321",
        pipeline.STATUS_DISPATCHED,
        None,
    ),
    (
        "short-noise",
        SHORT_QUERY,
        "Барахолка",
        None,
        pipeline.STATUS_FILTERED,
        pipeline.FILTER_REASON,
    ),
)

EXPECTED_COUNTS: Final[dict[str, int]] = {
    pipeline.STATUS_FILTERED: 4,
    pipeline.STATUS_DUPLICATE: 1,
    pipeline.STATUS_REJECTED_BY_LLM: 1,
    pipeline.STATUS_LLM_UNAVAILABLE: 1,
    pipeline.STATUS_DISPATCHED: 3,
}

#: Number of messages that survive Steps 1–2 and therefore reach the LLM.
EXPECTED_LLM_CALLS: Final[int] = 5


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


class Reporter:
    """Collects check results and prints them in a readable, aligned format."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failures: list[str] = []

    def section(self, title: str) -> None:
        print()
        print("-" * 76)
        print(title)
        print("-" * 76)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        """Record a single expectation and print its outcome immediately."""
        status = "[PASS]" if condition else "[FAIL]"
        line = f"  {status} {name}"
        if detail:
            line += f"  ->  {detail}"
        print(line)
        if condition:
            self.passed += 1
        else:
            self.failures.append(name)

    def summary(self) -> None:
        total = self.passed + len(self.failures)
        print()
        print("=" * 76)
        print(f" RESULTS: {self.passed}/{total} checks passed")
        if self.failures:
            print(f" FAILED CHECKS ({len(self.failures)}):")
            for name in self.failures:
                print(f"   - {name}")
        else:
            print(" All checks passed.")
        print("=" * 76)


# --------------------------------------------------------------------------- #
# Test suites
# --------------------------------------------------------------------------- #


async def test_batch(reporter: Reporter, db_path: str) -> None:
    """Feed 10 diverse messages through the pipeline and verify every pathway."""
    reporter.section("1. Batch of 10 messages through process_raw_message")

    classify_calls: list[str] = []
    dispatched: list[dict[str, Any]] = []

    async def fake_classify(text: str) -> dict[str, Any] | None:
        classify_calls.append(text)
        return CLASSIFIER_RESPONSES.get(text.strip())

    async def fake_send(
        lead_data: dict[str, Any],
        original_text: str,
        source_chat: str,
        msg_link: str | None = None,
        **_kwargs: Any,
    ) -> bool:
        dispatched.append(
            {
                "lead_data": lead_data,
                "original_text": original_text,
                "source_chat": source_chat,
                "msg_link": msg_link,
            }
        )
        return True

    counters: Counter[str] = Counter()
    results: list[tuple[str, dict[str, Any]]] = []

    with mock.patch.object(classifier, "classify_message", fake_classify), \
            mock.patch.object(notifier, "send_lead_alert", fake_send):
        for label, text, source, link, expected, reason in BATCH:
            result = await pipeline.process_raw_message(text, source, link, db_path)
            counters[result.get("status", "<missing>")] += 1
            results.append((label, result))

            correct_status = result.get("status") == expected
            correct_reason = reason is None or result.get("reason") == reason
            reporter.check(
                f"{label:<14} -> {expected}",
                correct_status and correct_reason,
                detail=str(result.get("status")) + (
                    f" (reason={result.get('reason')})"
                    if result.get("reason")
                    else ""
                ),
            )

    # -- Step distribution -------------------------------------------------- #
    reporter.section("2. Status counters")
    for status, expected in EXPECTED_COUNTS.items():
        reporter.check(
            f"counter {status}={expected}",
            counters[status] == expected,
            detail=f"got {counters[status]}",
        )
    reporter.check(
        "total processed = 10",
        sum(counters.values()) == 10,
        detail=f"got {sum(counters.values())}",
    )
    reporter.check(
        "no unexpected statuses",
        set(counters).issubset(set(EXPECTED_COUNTS)),
        detail=", ".join(sorted(set(counters) - set(EXPECTED_COUNTS))) or "none",
    )

    # -- Step gating -------------------------------------------------------- #
    reporter.section("3. Stage gating (no wasted paid calls)")
    reporter.check(
        f"LLM classifier called exactly {EXPECTED_LLM_CALLS} times",
        len(classify_calls) == EXPECTED_LLM_CALLS,
        detail=f"got {len(classify_calls)}",
    )
    reporter.check(
        "duplicate repost was NOT sent to the LLM",
        classify_calls.count(LEAD_AUTO.strip()) == 1,
        detail=f"seen {classify_calls.count(LEAD_AUTO.strip())}x",
    )
    reporter.check(
        "filtered messages never reached the LLM",
        not any(
            text in classify_calls
            for text in ("+", SHORT_QUERY, OFF_TOPIC, SPAM_AGENCY)
        ),
    )
    reporter.check(
        "notifier called exactly 3 times (leads only)",
        len(dispatched) == 3,
        detail=f"got {len(dispatched)}",
    )

    # -- Dispatch payload --------------------------------------------------- #
    reporter.section("4. Dispatch payload integrity")
    reporter.check(
        "only is_lead=True payloads were dispatched",
        all(item["lead_data"].get("is_lead") is True for item in dispatched),
    )
    reporter.check(
        "dispatched the three expected categories",
        {item["lead_data"]["category"] for item in dispatched}
        == {"Автоакция / Цены", "Логистика / Хранение", "Штрафы / Удержания"},
        detail=str(sorted(
            item["lead_data"]["category"] for item in dispatched
        )),
    )
    reporter.check(
        "source chat forwarded to notifier",
        all(item["source_chat"] for item in dispatched),
    )
    reporter.check(
        "auto-action lead carried its permalink",
        any(
            item["lead_data"]["category"] == "Автоакция / Цены"
            and item["msg_link"] == "https://t.me/wb_sellers_chat/12345"
            for item in dispatched
        ),
    )
    reporter.check(
        "dispatched results report delivered=True",
        all(
            result.get("delivered") is True
            for _, result in results
            if result.get("status") == pipeline.STATUS_DISPATCHED
        ),
    )

    # -- Deduplication cache state ----------------------------------------- #
    reporter.section("5. Deduplication cache state")
    again = await db_module.is_duplicate(LEAD_AUTO, db_path)
    reporter.check("auto-action lead recorded in the cache", again is True)


async def test_error_path(reporter: Reporter, db_path: str) -> None:
    """An unexpected exception must yield {"status": "error"} instead of raising."""
    reporter.section("6. Defensive error handling")

    async def exploding_classify(_text: str) -> dict[str, Any] | None:
        raise RuntimeError("simulated classifier crash")

    with mock.patch.object(classifier, "classify_message", exploding_classify):
        result = await pipeline.process_raw_message(
            ERROR_TEXT, "Error Test Chat", None, db_path
        )

    reporter.check(
        "unexpected exception -> status=error",
        result.get("status") == pipeline.STATUS_ERROR,
        detail=str(result.get("status")),
    )
    reporter.check(
        "error message propagated",
        "simulated classifier crash" in str(result.get("error", "")),
        detail=str(result.get("error")),
    )


async def test_llm_fallback(reporter: Reporter, db_path: str) -> None:
    """With ``llm_fallback=True`` an LLM outage dispatches a conservative lead."""
    reporter.section("7. Optional LLM fallback")

    async def null_classify(_text: str) -> dict[str, Any] | None:
        return None

    dispatched: list[dict[str, Any]] = []

    async def fake_send(
        lead_data: dict[str, Any],
        _original_text: str,
        _source_chat: str,
        _msg_link: str | None = None,
        **_kwargs: Any,
    ) -> bool:
        dispatched.append(lead_data)
        return True

    with mock.patch.object(classifier, "classify_message", null_classify), \
            mock.patch.object(notifier, "send_lead_alert", fake_send):
        result = await pipeline.process_raw_message(
            FALLBACK_TEXT, "Fallback Chat", None, db_path, llm_fallback=True
        )

    reporter.check(
        "fallback enabled -> dispatched",
        result.get("status") == pipeline.STATUS_DISPATCHED,
        detail=str(result.get("status")),
    )
    reporter.check(
        "fallback lead is low-confidence 'Другое'",
        bool(dispatched)
        and dispatched[0].get("category") == "Другое"
        and dispatched[0].get("confidence") == 1,
        detail=str(dispatched[0] if dispatched else None),
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run_suite() -> int:
    """Execute every integration suite and return a process exit code."""
    reporter = Reporter()

    print("=" * 76)
    print(" Sprint 5 — master pipeline end-to-end verification")
    print("=" * 76)

    with tempfile.TemporaryDirectory(prefix="lead_sniper_test_") as tmp:
        db_path = str(Path(tmp) / "pipeline_test.db")
        await db_module.init_db(db_path)
        await test_batch(reporter, db_path)
        await test_error_path(reporter, db_path)
        await test_llm_fallback(reporter, db_path)

    reporter.summary()
    if reporter.failures:
        print()
        print(
            f"[FAILURE] {len(reporter.failures)} check(s) failed: "
            + ", ".join(reporter.failures)
        )
        return 1

    print()
    print("[SUCCESS] Sprint 5 verification completed with 100% checks passing.")
    return 0


def main() -> int:
    """Synchronous wrapper so the script can be executed directly."""
    try:
        return asyncio.run(run_suite())
    except KeyboardInterrupt:
        print()
        print("[ABORTED] Interrupted by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
