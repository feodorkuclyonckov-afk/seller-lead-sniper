"""Sprint 1.3b — verification of the phone-number login flow in ``auth_test``.

Run with::

    python -m pytest test_auth_test.py -v

Only the pure, offline helpers are exercised here: phone-number validation,
environment reading, prompt reading and session-file verification. The
interactive code/password handshake is covered by the manual run of
``auth_test.py`` because it requires a real Telegram account, a login code and
possibly a cloud password.
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path

import pytest

import auth_test
from auth_test import (
    ConfigError,
    SessionError,
    normalize_phone,
    read_phone,
    read_session_name,
    verify_session_file,
)

# --------------------------------------------------------------------------- #
# normalize_phone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    ["+79991234567", "+7 999 123 45 67", "+7 (999) 123-45-67", "  +7-999-123-45-67  "],
)
def test_normalize_phone_accepts_formatted_e164(raw: str) -> None:
    """Whitespace, parentheses, dashes and spaces are stripped."""
    assert normalize_phone(raw) == "+79991234567"


@pytest.mark.parametrize(
    "raw",
    ["+12025550123", "+442071838750", "+8613800138000"],
)
def test_normalize_phone_keeps_other_countries(raw: str) -> None:
    """Non-Russian numbers pass through unchanged."""
    assert normalize_phone(raw) == raw


def test_normalize_phone_requires_leading_plus() -> None:
    """A number without the ``+`` prefix is rejected."""
    with pytest.raises(ConfigError, match="must start with '\\+'"):
        normalize_phone("79991234567")


@pytest.mark.parametrize("raw", ["", "   ", "+", "+ () -"])
def test_normalize_phone_rejects_empty(raw: str) -> None:
    """Blank input raises a configuration error."""
    with pytest.raises(ConfigError, match="empty"):
        normalize_phone(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "+7abc1234567",  # letters
        "+123456",  # too short
        "+7999123456789012",  # 17 digits — too long
        "+09991234567",  # country code cannot start with zero
    ],
)
def test_normalize_phone_rejects_invalid_numbers(raw: str) -> None:
    """Malformed numbers are rejected with a clear message."""
    with pytest.raises(ConfigError):
        normalize_phone(raw)


# --------------------------------------------------------------------------- #
# _read_line (prompt plumbing)
# --------------------------------------------------------------------------- #


def test_read_line_strips_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrounding whitespace is trimmed from the typed value."""
    monkeypatch.setattr(auth_test.sys, "stdin", io.StringIO("  +79991234567  \n"))
    assert auth_test._read_line("Phone: ") == "+79991234567"


def test_read_line_flushes_prompt_before_reading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The prompt is written to stdout even when it is not a TTY (piped run)."""
    monkeypatch.setattr(auth_test.sys, "stdin", io.StringIO(""))
    with pytest.raises(EOFError):
        auth_test._read_line("Phone: ")
    assert "Phone: " in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# read_phone
# --------------------------------------------------------------------------- #


def test_read_phone_returns_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset ``TG_PHONE`` yields an empty string so the CLI can prompt."""
    monkeypatch.delenv("TG_PHONE", raising=False)
    assert read_phone() == ""


def test_read_phone_normalizes_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured ``TG_PHONE`` is returned in E.164 form."""
    monkeypatch.setenv("TG_PHONE", " +7 999 123 45 67 ")
    assert read_phone() == "+79991234567"


def test_read_phone_rejects_invalid_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid ``TG_PHONE`` fails fast instead of hitting the network."""
    monkeypatch.setenv("TG_PHONE", "8-999-123-45-67")
    with pytest.raises(ConfigError):
        read_phone()


# --------------------------------------------------------------------------- #
# session name / session file verification
# --------------------------------------------------------------------------- #


def test_read_session_name_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TG_SESSION_NAME`` falls back to the module default."""
    monkeypatch.delenv("TG_SESSION_NAME", raising=False)
    assert read_session_name() == auth_test.DEFAULT_SESSION_NAME


def test_read_session_name_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``TG_SESSION_NAME`` wins over the default."""
    monkeypatch.setenv("TG_SESSION_NAME", "custom_session")
    assert read_session_name() == "custom_session"


def test_verify_session_file_returns_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty ``<name>.session`` file is accepted.

    The helper returns the path as written by Telethon (relative to the
    working directory), so the assertion checks the relative form and that the
    file really exists on disk.
    """
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "session_auth_test.session"
    session.write_bytes(b"SQLite format 3\x00")

    result = verify_session_file("session_auth_test")
    assert result == Path("session_auth_test.session")
    assert (tmp_path / result).is_file()


def test_verify_session_file_rejects_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing session file raises :class:`SessionError`."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SessionError, match="was not written"):
        verify_session_file("session_auth_test")


def test_verify_session_file_rejects_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-byte session file is treated as a failed authorization."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "session_auth_test.session").write_bytes(b"")
    with pytest.raises(SessionError, match="empty"):
        verify_session_file("session_auth_test")


# --------------------------------------------------------------------------- #
# pending login state (code requested in one run, submitted in the next)
# --------------------------------------------------------------------------- #


def test_pending_state_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving, loading and clearing the pending login state works."""
    monkeypatch.setattr(auth_test, "PENDING_STATE_FILE", tmp_path / ".auth_pending.json")

    assert auth_test.load_pending_state() == {}

    auth_test.save_pending_state("+79991234567", "deadbeefcafe")
    state = auth_test.load_pending_state()
    assert state["phone"] == "+79991234567"
    assert state["phone_code_hash"] == "deadbeefcafe"
    assert state["needs_password"] is False

    auth_test.save_pending_state("+79991234567", "deadbeefcafe", needs_password=True)
    assert auth_test.load_pending_state()["needs_password"] is True

    auth_test.clear_pending_state()
    assert auth_test.load_pending_state() == {}


def test_pending_state_ignores_corrupt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt state file is treated as "no pending login"."""
    path = tmp_path / ".auth_pending.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(auth_test, "PENDING_STATE_FILE", path)

    assert auth_test.load_pending_state() == {}


# --------------------------------------------------------------------------- #
# QR login (token rendered to qr.png)
# --------------------------------------------------------------------------- #


def test_render_qr_writes_png(tmp_path: Path) -> None:
    """``render_qr`` writes a real PNG file at the requested path."""
    target = tmp_path / "qr.png"

    result = auth_test.render_qr("tg://login?token=abc123", target)

    assert result == target
    assert target.is_file()
    assert target.read_bytes().startswith(b"\x89PNG")


def test_render_qr_is_atomic(tmp_path: Path) -> None:
    """No ``.tmp`` sibling is left behind after the image is rendered."""
    target = tmp_path / "qr.png"

    auth_test.render_qr("tg://login?token=abc123", target)

    assert not target.with_name("qr.png.tmp").exists()


def test_render_qr_requires_qrcode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``qrcode`` package surfaces as a configuration error."""
    monkeypatch.setattr(auth_test, "qrcode", None)

    with pytest.raises(ConfigError, match="qrcode"):
        auth_test.render_qr("tg://login?token=abc123", tmp_path / "qr.png")


def test_cleanup_qr_image_removes_file(tmp_path: Path) -> None:
    """The rendered image is deleted once the login completed."""
    target = tmp_path / "qr.png"
    target.write_bytes(b"\x89PNG")

    auth_test.cleanup_qr_image(target)

    assert not target.exists()


def test_cleanup_qr_image_missing_is_noop(tmp_path: Path) -> None:
    """Deleting an already-absent image is not an error."""
    auth_test.cleanup_qr_image(tmp_path / "qr.png")


class _StubQRLogin:
    """Minimal stand-in for Telethon's ``QRLogin``."""

    def __init__(self, client: "_StubClient", outcomes: list[str]) -> None:
        self._client = client
        self._outcomes = outcomes
        self.url = "tg://login?token=first"
        self.recreates = 0

    async def wait(self, timeout: float | None = None) -> object:
        outcome = self._outcomes.pop(0)
        if outcome == "timeout":
            raise asyncio.TimeoutError
        if outcome == "password":
            raise auth_test.SessionPasswordNeededError(request=None)
        self._client.authorized = True
        return object()

    async def recreate(self) -> None:
        self.recreates += 1
        self.url = f"tg://login?token=recreated{self.recreates}"


class _StubClient:
    """Minimal stand-in for ``TelegramClient`` covering the QR flow."""

    def __init__(self, outcomes: list[str]) -> None:
        self.authorized = False
        self.passwords: list[str | None] = []
        self.qr = _StubQRLogin(self, list(outcomes))

    async def qr_login(self) -> _StubQRLogin:
        return self.qr

    async def is_user_authorized(self) -> bool:
        return self.authorized

    async def sign_in(self, password: str | None = None) -> None:
        self.passwords.append(password)
        self.authorized = True


def test_qr_authenticate_refreshes_expired_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired token is recreated and ``qr.png`` is rewritten."""
    monkeypatch.setattr(auth_test, "QR_IMAGE_FILE", tmp_path / "qr.png")
    client = _StubClient(["timeout", "ok"])

    asyncio.run(auth_test.qr_authenticate(client))

    assert client.authorized
    assert client.qr.recreates == 1
    assert (tmp_path / "qr.png").is_file()


def test_qr_authenticate_uses_supplied_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2FA challenge raised by ``wait`` is completed with ``--password``."""
    monkeypatch.setattr(auth_test, "QR_IMAGE_FILE", tmp_path / "qr.png")
    client = _StubClient(["password"])

    asyncio.run(auth_test.qr_authenticate(client, password="secret"))

    assert client.passwords == ["secret"]
    assert client.authorized


def test_qr_authenticate_requires_password_when_non_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--password`` and no TTY the 2FA challenge fails clearly."""
    monkeypatch.setattr(auth_test, "QR_IMAGE_FILE", tmp_path / "qr.png")
    monkeypatch.setattr(auth_test.sys, "stdin", io.StringIO(""))
    client = _StubClient(["password"])

    with pytest.raises(auth_test.AuthError, match="--password"):
        asyncio.run(auth_test.qr_authenticate(client))


# --------------------------------------------------------------------------- #
# CLI flags
# --------------------------------------------------------------------------- #


def test_parse_args_qr_flag() -> None:
    """``--qr`` is exposed and defaults to off."""
    assert auth_test.parse_args([]).qr is False
    assert auth_test.parse_args(["--qr"]).qr is True


@pytest.mark.parametrize(
    "argv",
    [["--qr", "--phone", "+79991234567"], ["--qr", "--code", "12345"], ["--qr", "--request-code"]],
)
def test_parse_args_rejects_qr_combinations(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--qr`` cannot be mixed with the phone-code flags."""
    with pytest.raises(SystemExit) as excinfo:
        auth_test.parse_args(argv)
    assert excinfo.value.code == 2
    assert "--qr cannot be combined" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# proxy configuration
# --------------------------------------------------------------------------- #


def test_build_proxy_returns_none_when_host_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ``PROXY_HOST`` means a direct connection."""
    monkeypatch.delenv("PROXY_HOST", raising=False)
    assert auth_test.build_proxy() is None


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_build_proxy_returns_none_for_blank_host(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Empty or whitespace-only ``PROXY_HOST`` means a direct connection."""
    monkeypatch.setenv("PROXY_HOST", value)
    assert auth_test.build_proxy() is None


@pytest.mark.parametrize("value", ["127.0.0.1", "localhost", "LOCALHOST"])
def test_build_proxy_returns_none_for_loopback(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Loopback hosts are treated as "no proxy"."""
    monkeypatch.setenv("PROXY_HOST", value)
    assert auth_test.build_proxy() is None


def test_build_proxy_builds_dict_for_real_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real host produces a python-socks style proxy dict."""
    monkeypatch.setenv("PROXY_HOST", " 196.18.15.173 ")
    monkeypatch.setenv("PROXY_PORT", "8000")
    monkeypatch.setenv("PROXY_TYPE", "http")
    monkeypatch.delenv("PROXY_USER", raising=False)
    monkeypatch.delenv("PROXY_PASS", raising=False)
    # The proxy dict is built before Telethon imports python-socks; stub the
    # optional dependency so the test does not depend on the install extras.
    monkeypatch.setattr(auth_test, "python_socks", object())

    assert auth_test.build_proxy() == {
        "proxy_type": "http",
        "addr": "196.18.15.173",
        "port": 8000,
        "rdns": False,
    }


def test_load_environment_overrides_stale_proxy_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty ``PROXY_HOST`` in ``.env`` must win over a stale process value.

    This reproduces the reported bug: the shell still exported the old proxy
    address, and ``load_dotenv(override=False)`` left it in place, so the
    client silently kept using the proxy even though ``.env`` said direct.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("PROXY_HOST=\n", encoding="utf-8")
    monkeypatch.setattr(auth_test, "ENV_FILE", env_file)
    monkeypatch.setenv("PROXY_HOST", "196.18.15.173")

    asyncio.run(auth_test.load_environment())

    assert os.getenv("PROXY_HOST") == ""
    assert auth_test.build_proxy() is None
