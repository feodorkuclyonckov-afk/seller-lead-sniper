"""Pytest bridge for the standalone script-style verification suites.

``test_filters.py``, ``test_classifier.py`` and ``test_pipeline.py`` are
designed to run directly (``python test_filters.py``): their ``test_*``
functions accept an explicit ``Reporter`` (and sometimes a database path)
instead of relying on pytest fixtures, and they are not marked for
asynchronous execution.

This ``conftest.py`` lets those same functions run under pytest without editing
the test modules themselves:

* exposes ``reporter`` / ``db_path`` / ``tmp_root`` fixtures;
* marks their ``async def`` tests with ``pytest.mark.anyio`` so the event loop
  is provided by the anyio pytest plugin already used by ``test_notifier.py``.

``test_notifier.py`` defines its own module-level ``reporter`` fixture, which
takes precedence over the one here, so its behaviour is unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any, Iterator

import pytest

#: Script-style suites that need the pytest bridge defined in this module.
_BRIDGED_MODULES: frozenset[str] = frozenset(
    {"test_filters", "test_classifier", "test_pipeline"}
)


def _module_name(item: pytest.Item) -> str:
    """Return the module name owning ``item`` (empty when unavailable)."""
    module = getattr(item, "module", None)
    return getattr(module, "__name__", "")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Run async tests of the bridged suites through the anyio plugin."""
    del config
    for item in items:
        if _module_name(item) not in _BRIDGED_MODULES:
            continue
        function = getattr(item, "function", None)
        if function is not None and inspect.iscoroutinefunction(function):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture()
def reporter(request: pytest.FixtureRequest) -> Iterator[Any]:
    """Instantiate the ``Reporter`` defined by the requesting test module.

    The fixture asserts that no check failed after the test body finishes, so
    a failed ``reporter.check(...)`` turn into a failing pytest test.
    """
    module = request.module
    reporter_cls = getattr(module, "Reporter")
    instance = reporter_cls()
    yield instance
    failures = getattr(instance, "failures", [])
    assert not failures, (
        f"{len(failures)} check(s) failed in {module.__name__}: "
        + ", ".join(failures)
    )


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a module-scoped SQLite path shared by the dedupe checks.

    ``test_filters.py`` relies on earlier pain-point messages being recorded
    before ``test_duplicates`` resubmits one of them, so the path must live for
    the whole module.
    """
    directory = tmp_path_factory.mktemp("lead_sniper_pytest_db")
    return str(directory / "leads_cache.db")


@pytest.fixture()
def tmp_root(tmp_path: Any) -> Any:
    """Expose pytest's per-test temporary directory under the suite's name."""
    return tmp_path
