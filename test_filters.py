"""Sprint 2 — automated verification of the local filter and dedup cache.

Run with::

    .venv\\Scripts\\python.exe test_filters.py

The script is fully self-contained: it creates a throw-away SQLite database
inside a temporary directory, initialises it with :func:`db.init_db`, and then
exercises :func:`filters.passes_local_filter` and :func:`db.is_duplicate`
against a curated corpus of real-world style Russian messages.

Coverage
--------
* 3 genuine client requests — must pass the filter and be recorded once.
* 3 self-promotion spam messages — must be rejected by the stop-word gate.
* 2 length outliers (too short / too long) — must be rejected by the length
  gate.
* 2 duplicate submissions — the second one must be flagged by
  ``is_duplicate``.
* A few extra checks for hash normalisation and idempotent schema creation.

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

GENUINE_REQUESTS: Final[tuple[str, ...]] = (
    "Ищу карго из Китая в Россию, кто занимается доставкой под ключ? "
    "Нужен надёжный подрядчик.",
    "Посоветуйте, пожалуйста, кто возит сборные грузы из Гуанчжоу. "
    "Требуется карго с белым оформлением.",
    "Подскажите, кто может сделать таргетированную рекламу для "
    "интернет-магазина? Интересует маркетолог с опытом в e-commerce.",
)

SPAM_MESSAGES: Final[tuple[str, ...]] = (
    "Предлагаю услуги по продвижению в Telegram. Мои кейсы и портфолио "
    "в закрепе, пишите в личку.",
    "Возьму на ведение ваш проект, набираю учеников на курс по таргету. "
    "Бесплатная консультация.",
    "Продам базу клиентов для холодных рассылок. Свежие базы Telegram, "
    "приглашаю в канал.",
)

SHORT_MESSAGE: Final[str] = "Нужен карго."
LONG_MESSAGE: Final[str] = "Нужен карго из Китая в Москву, " * 40

DUPLICATE_ORIGINAL: Final[str] = (
    "Ищу подрядчика по карго из Китая в Москву, кто возит сборные грузы?"
)
# Same content, different case and punctuation — must hash identically.
DUPLICATE_REPOST: Final[str] = (
    "ищу подрядчика по карго из китая в москву кто возит сборные грузы"
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


def _first_match(text: str, words: tuple[str, ...]) -> str | None:
    """Return the first keyword found in ``text`` (already lowercased)."""
    for word in words:
        if word in text:
            return word
    return None


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

    folded_a = hash_text("Ищу КАРГО!!! из   Китая")
    folded_b = hash_text("ищу карго из китая")
    reporter.check(
        "case/punctuation/whitespace are normalised away",
        folded_a == folded_b,
        f"{folded_a[:12]}… == {folded_b[:12]}…",
    )

    digest = hash_text(GENUINE_REQUESTS[0])
    reporter.check(
        "digest is a 32-char lowercase hex MD5",
        len(digest) == 32 and all(c in "0123456789abcdef" for c in digest),
        digest,
    )

    different = hash_text(GENUINE_REQUESTS[0]) != hash_text(GENUINE_REQUESTS[1])
    reporter.check("distinct bodies produce distinct digests", different)

    reporter.check(
        "default DB path is 'leads_cache.db'",
        DEFAULT_DB_PATH == "leads_cache.db",
        DEFAULT_DB_PATH,
    )


async def test_genuine_requests(reporter: Reporter, db_path: str) -> None:
    """Real client inquiries must pass the filter and be recorded exactly once."""
    reporter.section("2. Genuine client requests (must pass filter + dedupe)")

    for index, message in enumerate(GENUINE_REQUESTS, start=1):
        label = f"request #{index}: {_shorten(message)}"

        passes = passes_local_filter(message)
        trigger = _first_match(message.lower(), TRIGGER_WORDS)
        blocked = _first_match(message.lower(), STOP_WORDS)
        reporter.check(
            f"{label} -> passes filter",
            passes and blocked is None,
            f"trigger={trigger!r}, stop={blocked!r}, len={len(message)}",
        )

        first_seen = await is_duplicate(message, db_path)
        reporter.check(
            f"{label} -> new for the cache (is_duplicate == False)",
            first_seen is False,
            f"is_duplicate={first_seen}",
        )


async def test_spam_messages(reporter: Reporter, db_path: str) -> None:
    """Self-promotion must be rejected by the stop-word gate, not by length."""
    reporter.section("3. Self-promotion / spam (must fail on stop words)")

    for index, message in enumerate(SPAM_MESSAGES, start=1):
        label = f"spam #{index}: {_shorten(message)}"

        passes = passes_local_filter(message)
        matched_stop = _first_match(message.lower(), STOP_WORDS)
        reporter.check(
            f"{label} -> rejected",
            passes is False and matched_stop is not None,
            f"stop={matched_stop!r}, len={len(message)}",
        )

        in_length_window = MIN_LENGTH <= len(message) <= MAX_LENGTH
        reporter.check(
            f"{label} -> length gate was not the reason",
            in_length_window,
            f"len={len(message)}",
        )


async def test_length_outliers(reporter: Reporter, db_path: str) -> None:
    """Messages outside [MIN_LENGTH, MAX_LENGTH] must be rejected."""
    reporter.section("4. Length outliers (must fail length check)")

    for name, message in (
        ("too short", SHORT_MESSAGE),
        ("too long", LONG_MESSAGE),
    ):
        label = f"{name}: {_shorten(message)}"

        passes = passes_local_filter(message)
        reporter.check(
            f"{label} -> rejected ({_length_gate(message)})",
            passes is False,
            f"passes={passes}, len={len(message)}",
        )

        # These samples contain triggers, proving the length gate fired first
        # rather than the trigger gate.
        has_trigger = _first_match(message.lower(), TRIGGER_WORDS) is not None
        reporter.check(
            f"{label} -> contains a trigger word (length was decisive)",
            has_trigger,
        )


async def test_duplicates(reporter: Reporter, db_path: str) -> None:
    """The same body (even reposted with different punctuation) is reported once."""
    reporter.section("5. Duplicate detection (must be caught by is_duplicate)")

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

    # Pair 2: resubmitting an earlier genuine request must also be caught.
    again = await is_duplicate(GENUINE_REQUESTS[0], db_path)
    reporter.check(
        "genuine request #1 resubmitted later -> duplicate",
        again is True,
        f"is_duplicate={again}",
    )


async def test_schema_idempotency(reporter: Reporter, tmp_root: Path) -> None:
    """``init_db`` must be safe to call repeatedly and self-heal on demand."""
    reporter.section("6. Schema handling")

    path = str(tmp_root / "idempotent_cache.db")
    await init_db(path)
    await init_db(path)
    reporter.check("init_db() is idempotent", Path(path).is_file(), path)

    healed = str(tmp_root / "never_initialised.db")
    new_message = await is_duplicate(
        "Ищу поставщика карго для своего магазина, подскажите кого-нибудь.",
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
    print(" Sprint 2 — local keyword filter & SQLite deduplication verification")
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

        await test_genuine_requests(reporter, db_path)
        await test_spam_messages(reporter, db_path)
        await test_length_outliers(reporter, db_path)
        await test_duplicates(reporter, db_path)
        await test_schema_idempotency(reporter, Path(tmp))

    reporter.summary()
    assert not reporter.failures, (
        f"{len(reporter.failures)} check(s) failed: " + ", ".join(reporter.failures)
    )
    print()
    print("[SUCCESS] Sprint 2 verification completed with 100% checks passing.")
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
