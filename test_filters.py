"""Local pre-filter & dedup verification for marketplace seller pain points.

Run with::

    .venv\\Scripts\\python.exe test_filters.py

The script is fully self-contained: it creates a throw-away SQLite database
inside a temporary directory, initialises it with :func:`db.init_db`, and then
exercises :func:`filters.passes_local_filter` and :func:`db.is_duplicate`
against a curated corpus of realistic Russian messages from Wildberries /
Ozon seller chats.

Coverage
--------
* 4 seller pain-point messages (forced auto-promotions, negative unit
  economics, warehouse coefficient x5, dimension fines) — must pass the
  filter, matching the required trigger stem.
* 4 agency / spam messages — must be rejected by the stop-word gate while
  still fitting inside the length window.
* 2 casual chat messages — must be rejected because no trigger stem occurs.
* 2 length outliers (too short / too long) — must be rejected by the length
  gate; the long one deliberately contains triggers to prove the length gate
  fired first.
* 4 duplicate submissions — the second one must be flagged by
  ``is_duplicate``.
* Extra checks for hash normalisation, schema idempotency and default paths.

Exit code ``0`` means every check passed; ``1`` means at least one failed.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Final

from db import DEFAULT_DB_PATH, hash_text, init_db, is_duplicate
from filters import MAX_LENGTH, MIN_LENGTH, STOP_WORDS, TRIGGER_WORDS, passes_local_filter

# --------------------------------------------------------------------------- #
# Sample corpus
# --------------------------------------------------------------------------- #

#: ``(message, required_trigger_stem)`` — every message must survive the
#: filter *and* contain the given stem, which must itself be listed in
#: :data:`filters.TRIGGER_WORDS`.
POSITIVE_CASES: Final[tuple[tuple[str, str], ...]] = (
    (
        "Включили принудительную автоакцию на весь ассортимент, скидки рисуют "
        "сами, маржа уже в минус. Кто сталкивался, как отключить?",
        "автоакци",
    ),
    (
        "Товар после скидки WB ушёл в минус: себестоимость выше выплаты на 300 "
        "рублей. Как считаете юнит-экономику в таких условиях?",
        "себестоимост",
    ),
    (
        "Озон выставил коэффициент склада х5 на приемку, логистика съела всю "
        "прибыль за месяц. Это вообще законно, кто оспаривал?",
        "коэффициент",
    ),
    (
        "Прилетел штраф за завышенные габариты после перемера, удержание 40 "
        "тысяч из отчёта. Как доказать, что замер был неверный?",
        "штраф",
    ),
)

#: ``(message, expected_stop_word)`` — must be rejected by the stop-word gate.
STOP_WORD_CASES: Final[tuple[tuple[str, str], ...]] = (
    (
        "Предлагаю услуги по продвижению карточек на WB под ключ, пишите в "
        "личку, покажу кейсы.",
        "предлагаю услуги",
    ),
    (
        "Выведу в топ Ozon за месяц, оформление карточек и инфографика в "
        "подарок к ведению кабинета.",
        "выведу в топ",
    ),
    (
        "Накрутка отзывов и выкупов, байер из Китая, фулфилмент Москва — "
        "пишите в лс, работаем с 2019 года.",
        "накрутк",
    ),
    (
        "Курс по выходу в топ маркетплейсов: продвижение, инфографика, разбор "
        "личного кабинета. Осталось два места.",
        "курс по",
    ),
)

#: Casual chatter — long enough, no stop words, but no pain-point trigger.
CASUAL_MESSAGES: Final[tuple[str, ...]] = (
    "Привет всем! Кто сегодня тоже сидит без заказов из-за падения спроса на "
    "маркетплейсах?",
    "Привет всем, как настроение? У нас в Новосибирске уже выпал снег, а "
    "отчёты ещё не сдали.",
)

SHORT_MESSAGE: Final[str] = "Где склад?"

#: Deliberately longer than :data:`filters.MAX_LENGTH` while containing
#: triggers, so a rejection can only come from the length gate.
LONG_MESSAGE: Final[str] = (
    "Принудительная автоакция снизила маржу, товар ушёл в минус. " * 30
)

DUPLICATE_ORIGINAL: Final[str] = (
    "Коэффициент логистики вырос до х5, товар ушел в минус, подскажите, как "
    "быть с поставками?"
)
# Same content, different case and punctuation — must hash identically.
DUPLICATE_REPOST: Final[str] = (
    "коэффициент логистики вырос до х5 товар ушел в минус подскажите как "
    "быть с поставками"
)


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


def _shorten(text: str, limit: int = 64) -> str:
    """Return a single-line, length-capped preview of ``text``."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _first_match(text_lower: str, words: tuple[str, ...]) -> str | None:
    """Return the first keyword found in ``text_lower``, else ``None``."""
    for word in words:
        if word in text_lower:
            return word
    return None


def _in_length_window(text: str) -> bool:
    """Whether ``text`` satisfies the ``[MIN_LENGTH, MAX_LENGTH]`` window."""
    return MIN_LENGTH <= len(text) <= MAX_LENGTH


def _length_gate(text: str) -> str:
    """Describe which side of the length window ``text`` falls outside of."""
    if len(text) < MIN_LENGTH:
        return f"too short ({len(text)} < {MIN_LENGTH})"
    return f"too long ({len(text)} > {MAX_LENGTH})"


# --------------------------------------------------------------------------- #
# Test suites
# --------------------------------------------------------------------------- #


def test_hash_normalisation(reporter: Reporter) -> None:
    """``hash_text`` must ignore case, punctuation and repeated whitespace."""
    reporter.section("1. Hash normalisation (db.hash_text)")

    folded_a = hash_text("Коэффициент ЛОГИСТИКИ х5!!! ушёл в   минус")
    folded_b = hash_text("коэффициент логистики х5 ушёл в минус")
    reporter.check(
        "case/punctuation/whitespace are normalised away",
        folded_a == folded_b,
        f"{folded_a[:12]}… == {folded_b[:12]}…",
    )

    digest = hash_text(POSITIVE_CASES[0][0])
    reporter.check(
        "digest is a 32-char lowercase hex MD5",
        len(digest) == 32 and all(c in "0123456789abcdef" for c in digest),
        digest,
    )

    different = (
        hash_text(POSITIVE_CASES[0][0]) != hash_text(POSITIVE_CASES[1][0])
    )
    reporter.check("distinct bodies produce distinct digests", different)

    reporter.check(
        "default DB path is 'leads_cache.db'",
        DEFAULT_DB_PATH == "leads_cache.db",
        DEFAULT_DB_PATH,
    )


async def test_positive_cases(reporter: Reporter, db_path: str) -> None:
    """Seller pain-point messages must pass the filter and be recorded once."""
    reporter.section("2. Seller pain points (must pass filter + dedupe)")

    for index, (message, required_stem) in enumerate(POSITIVE_CASES, start=1):
        label = f"pain #{index}: {_shorten(message)}"
        lowered = message.lower()

        passes = passes_local_filter(message)
        trigger = _first_match(lowered, TRIGGER_WORDS)
        blocked = _first_match(lowered, STOP_WORDS)
        reporter.check(
            f"{label} -> passes filter",
            passes and blocked is None,
            f"trigger={trigger!r}, stop={blocked!r}, len={len(message)}",
        )

        reporter.check(
            f"{label} -> length inside window",
            _in_length_window(message),
            f"len={len(message)} in [{MIN_LENGTH}, {MAX_LENGTH}]",
        )

        reporter.check(
            f"{label} -> stem {required_stem!r} matched",
            required_stem in TRIGGER_WORDS and required_stem in lowered,
            f"trigger={trigger!r}",
        )

        first_seen = await is_duplicate(message, db_path)
        reporter.check(
            f"{label} -> new for the cache (is_duplicate == False)",
            first_seen is False,
            f"is_duplicate={first_seen}",
        )


def test_stop_word_cases(reporter: Reporter) -> None:
    """Agency offers must be rejected by the stop-word gate, not by length."""
    reporter.section("3. Agency / spam offers (must fail on stop words)")

    for index, (message, expected_stop) in enumerate(STOP_WORD_CASES, start=1):
        label = f"spam #{index}: {_shorten(message)}"
        lowered = message.lower()

        passes = passes_local_filter(message)
        matched_stop = _first_match(lowered, STOP_WORDS)
        reporter.check(
            f"{label} -> rejected",
            passes is False and matched_stop is not None,
            f"stop={matched_stop!r}, len={len(message)}",
        )

        reporter.check(
            f"{label} -> expected stop word {expected_stop!r} present",
            expected_stop in STOP_WORDS and expected_stop in lowered,
            f"stop={matched_stop!r}",
        )

        reporter.check(
            f"{label} -> length gate was not the reason",
            _in_length_window(message),
            f"len={len(message)}",
        )


def test_casual_chat_cases(reporter: Reporter) -> None:
    """Casual chatter lacks any pain-point trigger and must be dropped."""
    reporter.section("4. Casual chat (must fail: no trigger word)")

    for index, message in enumerate(CASUAL_MESSAGES, start=1):
        label = f"chat #{index}: {_shorten(message)}"
        lowered = message.lower()

        passes = passes_local_filter(message)
        trigger = _first_match(lowered, TRIGGER_WORDS)
        blocked = _first_match(lowered, STOP_WORDS)
        reporter.check(
            f"{label} -> rejected",
            passes is False and trigger is None,
            f"trigger={trigger!r}, stop={blocked!r}, len={len(message)}",
        )

        reporter.check(
            f"{label} -> length and stop gates were not the reason",
            _in_length_window(message) and blocked is None,
            f"len={len(message)}",
        )


def test_length_outliers(reporter: Reporter) -> None:
    """Messages outside [MIN_LENGTH, MAX_LENGTH] must be rejected."""
    reporter.section("5. Length outliers (must fail length check)")

    for name, message in (
        ("too short", SHORT_MESSAGE),
        ("too long", LONG_MESSAGE),
    ):
        label = f"{name}: {_shorten(message)}"

        passes = passes_local_filter(message)
        outside = not _in_length_window(message)
        reporter.check(
            f"{label} -> rejected ({_length_gate(message)})",
            passes is False and outside,
            f"passes={passes}, len={len(message)}",
        )

    # The long sample contains triggers, proving that the length gate — not
    # the trigger gate — was decisive for it.
    reporter.check(
        "too long -> contains a trigger word (length was decisive)",
        _first_match(LONG_MESSAGE.lower(), TRIGGER_WORDS) is not None,
        f"trigger={_first_match(LONG_MESSAGE.lower(), TRIGGER_WORDS)!r}",
    )


async def test_duplicates(reporter: Reporter, db_path: str) -> None:
    """The same body (even reposted with different punctuation) is reported once."""
    reporter.section("6. Duplicate detection (must be caught by is_duplicate)")

    # Pair 1: a fresh message and its punctuation/case-shifted repost.
    reporter.check(
        "duplicate pair passes the local filter",
        passes_local_filter(DUPLICATE_ORIGINAL),
        "so only the cache can stop the repost",
    )

    first = await is_duplicate(DUPLICATE_ORIGINAL, db_path)
    reporter.check(
        "duplicate pair, submission #1 -> not a duplicate",
        first is False,
        f"is_duplicate={first}",
    )

    second = await is_duplicate(DUPLICATE_REPOST, db_path)
    reporter.check(
        "duplicate pair, submission #2 (repost) -> duplicate",
        second is True,
        f"is_duplicate={second}",
    )

    # Pair 2: resubmitting an earlier pain-point message must also be caught.
    again = await is_duplicate(POSITIVE_CASES[0][0], db_path)
    reporter.check(
        "pain message #1 resubmitted later -> duplicate",
        again is True,
        f"is_duplicate={again}",
    )


async def test_schema_idempotency(reporter: Reporter, tmp_root: Path) -> None:
    """``init_db`` must be safe to call repeatedly and self-heal on demand."""
    reporter.section("7. Schema handling")

    path = str(tmp_root / "idempotent_cache.db")
    await init_db(path)
    await init_db(path)
    reporter.check("init_db() is idempotent", Path(path).is_file(), path)

    healed = str(tmp_root / "never_initialised.db")
    new_message = await is_duplicate(
        "Штраф за габариты удержали из отчёта реализации, как оспорить удержание?",
        healed,
    )
    reporter.check(
        "is_duplicate() works without a prior init_db() call",
        new_message is False and Path(healed).is_file(),
        "schema auto-created",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run_suite() -> int:
    """Execute every test suite and return a process exit code."""
    reporter = Reporter()

    print("=" * 76)
    print(" Local filter & SQLite deduplication verification (seller pain points)")
    print("=" * 76)

    test_hash_normalisation(reporter)

    with tempfile.TemporaryDirectory(prefix="lead_sniper_sprint2_") as tmp:
        db_path = str(Path(tmp) / "leads_cache.db")
        await init_db(db_path)
        reporter.section("0. Database bootstrap")
        reporter.check(
            "init_db() created the cache file",
            Path(db_path).is_file(),
            db_path,
        )

        await test_positive_cases(reporter, db_path)
        test_stop_word_cases(reporter)
        test_casual_chat_cases(reporter)
        test_length_outliers(reporter)
        await test_duplicates(reporter, db_path)
        await test_schema_idempotency(reporter, Path(tmp))

    reporter.summary()
    assert not reporter.failures, (
        f"{len(reporter.failures)} check(s) failed: " + ", ".join(reporter.failures)
    )
    print()
    print("[SUCCESS] Filter verification completed with 100% checks passing.")
    return 0


def main() -> int:
    """Synchronous wrapper so the script can be executed directly."""
    try:
        return asyncio.run(run_suite())
    except AssertionError as exc:
        print()
        print(f"[FAILURE] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
