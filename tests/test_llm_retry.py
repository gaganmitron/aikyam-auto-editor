"""_with_retry (planner_llm.py): retries transient Claude API errors (429/503/529, or a "rate"/
"overloaded" message) with backoff, but lets any other error through immediately -- so a real
prompt/schema bug still falls back to the deterministic plan on the first try, not the third."""
import pytest
from aikyam_video.planner_llm import _with_retry


class _Transient(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def test_retries_until_success_on_a_transient_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Transient(429)
        return "ok"

    assert _with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_gives_up_after_the_last_attempt(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(_Transient):
        _with_retry(lambda: (_ for _ in ()).throw(_Transient(503)), attempts=2)


def test_a_non_transient_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("bad schema")

    with pytest.raises(ValueError):
        _with_retry(bad_request)
    assert calls["n"] == 1
