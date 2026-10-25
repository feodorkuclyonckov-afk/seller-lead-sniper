"""Sprint 3 — verification of the async LLM lead classifier.

Run with::

    .venv\\Scripts\\python.exe test_classifier.py

The suite has two parts:

* **Offline (always runs).** A fake ``AsyncOpenAI``-compatible client feeds
  canned responses and exceptions into :func:`classifier.classify_message`,
  covering JSON parsing, schema normalisation, retries and failure handling.
  No network access and no API key are required.
* **Live (optional).** When ``LLM_API_KEY`` is present in ``.env`` (or the
  process environment), one real request is sent to the configured endpoint to
  prove the credentials and model work end to end.

Exit code ``0`` means every executed check passed; ``1`` means at least one
failed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Final, Iterator
from unittest import mock

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

import classifier
from classifier import (
    CATEGORIES,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMConfig,
    classify_message,
    load_llm_config,
    read_llm_config,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

GENUINE_REQUEST: Final[str] = (
    "Ищу карго из Китая в Москву, кто занимается доставкой под ключ? "
    "Нужен надёжный подрядчик."
)

TEST_CONFIG: Final[LLMConfig] = LLMConfig(
    api_key="test-key",
    base_url="https://llm.example.invalid",
    model="test-model",
)

_REQUEST: Final[httpx.Request] = httpx.Request(
    "POST", "https://llm.example.invalid/chat/completions"
)


def _valid_payload() -> dict[str, Any]:
    """Return a well-formed classifier payload."""
    return {
        "is_lead": True,
        "category": "Карго / Логистика",
        "summary": "Поиск карго из Китая в Москву",
        "confidence": 9,
    }


def _valid_json() -> str:
    return json.dumps(_valid_payload(), ensure_ascii=False)


def _make_response(content: str | None) -> SimpleNamespace:
    """Build a minimal object shaped like an OpenAI chat completion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    """Stand-in for ``client.chat.completions`` that records every call."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._handler(kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    """Minimal ``AsyncOpenAI``-compatible client used by the offline tests."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.completions = FakeCompletions(handler)
        self.chat = SimpleNamespace(completions=self.completions)


def _static(content: str | None) -> Callable[[dict[str, Any]], Any]:
    """Return a handler that always yields ``content``."""
    return lambda _kwargs: _make_response(content)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_REQUEST)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=_REQUEST)


def _rate_limit_error() -> RateLimitError:
    response = httpx.Response(429, request=_REQUEST)
    return RateLimitError("rate limited", response=response, body=None)


def _status_error(status: int) -> APIStatusError:
    response = httpx.Response(status, request=_REQUEST)
    return APIStatusError("bad request", response=response, body=None)


@contextlib.contextmanager
def _llm_env(values: dict[str, str] | None = None) -> Iterator[None]:
    """Temporarily replace the ``LLM_*`` environment variables."""
    keys = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    if values:
        os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


# --------------------------------------------------------------------------- #
# Test suites
# --------------------------------------------------------------------------- #


async def test_configuration(reporter: Reporter) -> None:
    """Defaults must be clean and environment overrides must be stripped."""
    reporter.section("1. Configuration loading (defaults + overrides)")

    with _llm_env():
        cfg = await load_llm_config(env_file=Path("definitely-missing.env"))

    reporter.check(
        "missing key leaves the classifier disabled",
        cfg.api_key == "" and cfg.is_configured is False,
        f"api_key={cfg.api_key!r}",
    )
    reporter.check(
        "base URL falls back to the DeepSeek default",
        cfg.base_url == DEFAULT_BASE_URL,
        cfg.base_url,
    )
    reporter.check(
        "model falls back to deepseek-chat",
        cfg.model == DEFAULT_MODEL,
        cfg.model,
    )

    with _llm_env(
        {
            "LLM_API_KEY": "  secret-key  ",
            "LLM_BASE_URL": "  https://example.test/v1  ",
            "LLM_MODEL": "  custom-model  ",
        }
    ):
        overridden = read_llm_config()

    reporter.check(
        "environment overrides are read and stripped",
        overridden
        == LLMConfig(
            api_key="secret-key",
            base_url="https://example.test/v1",
            model="custom-model",
        ),
        f"{overridden.api_key!r} @ {overridden.base_url!r} / {overridden.model!r}",
    )
    reporter.check(
        "is_configured is True once a key is present",
        overridden.is_configured is True,
    )


async def test_happy_path(reporter: Reporter) -> None:
    """A valid response must be normalised and the request must be correct."""
    reporter.section("2. Happy path (valid JSON + request contract)")

    client = FakeClient(_static(_valid_json()))
    result = await classify_message(
        GENUINE_REQUEST, client=client, config=TEST_CONFIG, max_retries=0
    )

    reporter.check(
        "valid JSON is parsed into the canonical schema",
        result == _valid_payload(),
        repr(result),
    )

    calls = client.completions.calls
    reporter.check("exactly one API call was made", len(calls) == 1, str(len(calls)))

    if calls:
        kwargs = calls[0]
        reporter.check(
            "request uses the configured model",
            kwargs.get("model") == TEST_CONFIG.model,
            repr(kwargs.get("model")),
        )
        reporter.check(
            "request enforces JSON object output",
            kwargs.get("response_format") == {"type": "json_object"},
            repr(kwargs.get("response_format")),
        )
        reporter.check(
            "request is deterministic (temperature=0)",
            kwargs.get("temperature") == 0.0,
            repr(kwargs.get("temperature")),
        )
        messages = kwargs.get("messages") or []
        reporter.check(
            "system prompt is sent first",
            len(messages) == 2
            and messages[0].get("role") == "system"
            and messages[0].get("content") == classifier.SYSTEM_PROMPT,
        )
        reporter.check(
            "user message contains the original text",
            len(messages) == 2
            and messages[1].get("role") == "user"
            and GENUINE_REQUEST in messages[1].get("content", ""),
        )


async def test_parsing_and_normalisation(reporter: Reporter) -> None:
    """Malformed-but-recoverable payloads must be coerced, not crash."""
    reporter.section("3. JSON parsing & schema normalisation")

    long_summary = "один два три четыре пять шесть семь восемь девять десять"
    cases: list[tuple[str, str | None, dict[str, Any] | None]] = [
        (
            "markdown-fenced JSON is unwrapped",
            "```json\n" + _valid_json() + "\n```",
            _valid_payload(),
        ),
        (
            "JSON surrounded by prose is extracted",
            "Вот результат: " + _valid_json() + " Готово.",
            _valid_payload(),
        ),
        (
            "unknown category falls back to 'Другое'",
            json.dumps(
                {
                    "is_lead": True,
                    "category": "Юристы",
                    "summary": "Нужен юрист по договорам",
                    "confidence": 6,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": True,
                "category": "Другое",
                "summary": "Нужен юрист по договорам",
                "confidence": 6,
            },
        ),
        (
            "category matching is case-insensitive",
            json.dumps(
                {
                    "is_lead": True,
                    "category": "карго / логистика",
                    "summary": "Доставка груза из Китая",
                    "confidence": 8,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": True,
                "category": "Карго / Логистика",
                "summary": "Доставка груза из Китая",
                "confidence": 8,
            },
        ),
        (
            "confidence above 10 is clamped",
            json.dumps(
                {
                    "is_lead": True,
                    "category": "Фулфилмент",
                    "summary": "Сборка заказов на складе",
                    "confidence": 42,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": True,
                "category": "Фулфилмент",
                "summary": "Сборка заказов на складе",
                "confidence": 10,
            },
        ),
        (
            "confidence below 1 is clamped",
            json.dumps(
                {
                    "is_lead": False,
                    "category": "Другое",
                    "summary": "Просто сообщение в чате",
                    "confidence": -5,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": False,
                "category": "Другое",
                "summary": "Просто сообщение в чате",
                "confidence": 1,
            },
        ),
        (
            "numeric-string confidence is accepted",
            json.dumps(
                {
                    "is_lead": True,
                    "category": "Маркетинг / Трафик",
                    "summary": "Нужен таргетолог для магазина",
                    "confidence": "7",
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": True,
                "category": "Маркетинг / Трафик",
                "summary": "Нужен таргетолог для магазина",
                "confidence": 7,
            },
        ),
        (
            "summary is capped at 8 words",
            json.dumps(
                {
                    "is_lead": True,
                    "category": "Дизайн / Контент",
                    "summary": long_summary,
                    "confidence": 5,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": True,
                "category": "Дизайн / Контент",
                "summary": " ".join(long_summary.split()[:8]),
                "confidence": 5,
            },
        ),
        (
            "string 'true' is coerced to bool",
            json.dumps(
                {
                    "is_lead": "true",
                    "category": "Фулфилмент",
                    "summary": "Нужен фулфилмент для маркетплейса",
                    "confidence": 8,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": True,
                "category": "Фулфилмент",
                "summary": "Нужен фулфилмент для маркетплейса",
                "confidence": 8,
            },
        ),
        (
            "string 'нет' is coerced to False",
            json.dumps(
                {
                    "is_lead": "нет",
                    "category": "Другое",
                    "summary": "Реклама курса по маркетингу",
                    "confidence": 2,
                },
                ensure_ascii=False,
            ),
            {
                "is_lead": False,
                "category": "Другое",
                "summary": "Реклама курса по маркетингу",
                "confidence": 2,
            },
        ),
        ("invalid JSON returns None", "not json at all", None),
        ("JSON array returns None", "[1, 2, 3]", None),
        (
            "missing summary returns None",
            json.dumps(
                {"is_lead": True, "category": "Другое", "confidence": 5},
                ensure_ascii=False,
            ),
            None,
        ),
        (
            "missing is_lead returns None",
            json.dumps(
                {
                    "category": "Другое",
                    "summary": "Непонятное сообщение",
                    "confidence": 5,
                },
                ensure_ascii=False,
            ),
            None,
        ),
        ("empty content returns None", None, None),
        ("blank content returns None", "   ", None),
    ]

    for name, content, expected in cases:
        client = FakeClient(_static(content))
        result = await classify_message(
            GENUINE_REQUEST, client=client, config=TEST_CONFIG, max_retries=0
        )
        reporter.check(
            name,
            result == expected,
            f"got {result!r}",
        )


async def test_api_failures(reporter: Reporter) -> None:
    """API errors must be swallowed and reported as ``None``."""
    reporter.section("4. API failure handling (timeouts, limits, bad status)")

    failure_cases: list[tuple[str, Callable[[], BaseException], int]] = [
        ("timeout returns None", _timeout_error, 1),
        ("rate limit returns None", _rate_limit_error, 1),
        ("connection error returns None", _connection_error, 1),
        ("HTTP 400 returns None", lambda: _status_error(400), 1),
        ("unexpected exception returns None", lambda: RuntimeError("boom"), 1),
    ]

    for name, factory, expected_calls in failure_cases:
        client = FakeClient(lambda _kwargs, f=factory: f())
        result = await classify_message(
            GENUINE_REQUEST, client=client, config=TEST_CONFIG, max_retries=0
        )
        reporter.check(
            name,
            result is None and len(client.completions.calls) == expected_calls,
            f"result={result!r}, calls={len(client.completions.calls)}",
        )

    # Transient failure followed by success: the retry must recover.
    attempts = {"count": 0}

    def flaky(_kwargs: dict[str, Any]) -> Any:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _timeout_error()
        return _make_response(_valid_json())

    client = FakeClient(flaky)
    with mock.patch.object(classifier, "RETRY_BASE_DELAY", 0.0):
        result = await classify_message(
            GENUINE_REQUEST, client=client, config=TEST_CONFIG, max_retries=1
        )
    reporter.check(
        "transient timeout is retried and then succeeds",
        result == _valid_payload() and len(client.completions.calls) == 2,
        f"result={result!r}, calls={len(client.completions.calls)}",
    )

    # Exhausted retries must stop after the configured number of attempts.
    client = FakeClient(lambda _kwargs: _connection_error())
    with mock.patch.object(classifier, "RETRY_BASE_DELAY", 0.0):
        result = await classify_message(
            GENUINE_REQUEST, client=client, config=TEST_CONFIG, max_retries=2
        )
    reporter.check(
        "exhausted retries stop after max_retries + 1 attempts",
        result is None and len(client.completions.calls) == 3,
        f"result={result!r}, calls={len(client.completions.calls)}",
    )


async def test_guards(reporter: Reporter) -> None:
    """Empty input and missing credentials must short-circuit without a call."""
    reporter.section("5. Input & configuration guards")

    client = FakeClient(_static(_valid_json()))
    empty = await classify_message(
        "   \n\t ", client=client, config=TEST_CONFIG, max_retries=0
    )
    reporter.check(
        "blank message returns None without calling the API",
        empty is None and not client.completions.calls,
        f"result={empty!r}, calls={len(client.completions.calls)}",
    )

    unconfigured = LLMConfig(api_key="", base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL)
    client = FakeClient(_static(_valid_json()))
    skipped = await classify_message(
        GENUINE_REQUEST, client=client, config=unconfigured, max_retries=0
    )
    reporter.check(
        "missing API key returns None without calling the API",
        skipped is None and not client.completions.calls,
        f"result={skipped!r}, calls={len(client.completions.calls)}",
    )

    raised: BaseException | None = None
    try:
        await classify_message(123, client=client, config=TEST_CONFIG)  # type: ignore[arg-type]
    except BaseException as exc:  # noqa: BLE001 - we assert the exact type below
        raised = exc
    reporter.check(
        "non-string input raises TypeError",
        isinstance(raised, TypeError),
        type(raised).__name__ if raised else "no exception",
    )


async def test_live_api(reporter: Reporter) -> None:
    """Optionally hit the real endpoint when ``LLM_API_KEY`` is configured."""
    reporter.section("6. Live API check (optional)")

    config = await load_llm_config()
    if not config.is_configured:
        print(
            "  [SKIP] LLM_API_KEY is not set in .env or the environment — "
            "offline checks only."
        )
        return

    print(
        f"  [INFO] Using {config.base_url} with model {config.model!r}."
    )
    try:
        result = await classify_message(
            GENUINE_REQUEST, config=config, max_retries=1
        )
    finally:
        await classifier.aclose()

    reporter.check(
        "live API returns a parsed dict",
        isinstance(result, dict),
        repr(result),
    )
    if not isinstance(result, dict):
        return

    reporter.check(
        "live result has exactly the four schema keys",
        set(result) == {"is_lead", "category", "summary", "confidence"},
        repr(sorted(result)),
    )
    reporter.check(
        "live is_lead is a bool",
        isinstance(result.get("is_lead"), bool),
        repr(result.get("is_lead")),
    )
    reporter.check(
        "live category is one of the allowed values",
        result.get("category") in CATEGORIES,
        repr(result.get("category")),
    )
    reporter.check(
        "live summary is a non-empty string",
        isinstance(result.get("summary"), str) and bool(result["summary"].strip()),
        repr(result.get("summary")),
    )
    reporter.check(
        "live confidence is an int in 1..10",
        isinstance(result.get("confidence"), int)
        and 1 <= result["confidence"] <= 10,
        repr(result.get("confidence")),
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run_suite() -> int:
    """Execute every test suite and return a process exit code."""
    reporter = Reporter()

    print("=" * 76)
    print(" Sprint 3 — async LLM lead classifier verification")
    print("=" * 76)

    await test_configuration(reporter)
    await test_happy_path(reporter)
    await test_parsing_and_normalisation(reporter)
    await test_api_failures(reporter)
    await test_guards(reporter)
    await test_live_api(reporter)

    reporter.summary()
    assert not reporter.failures, (
        f"{len(reporter.failures)} check(s) failed: "
        + ", ".join(reporter.failures)
    )
    print()
    print("[SUCCESS] Sprint 3 verification completed with 100% checks passing.")
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
