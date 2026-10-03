"""Backend path completion must arrive at ``terminal_tool`` pre-confirmed.

The composer's remote-backend directory listing wraps a fixed read-only script in
``sh -c``, which the dangerous-command guard flags as "shell command via -c/-lc
flag". Under smart approvals that verdict fires an auxiliary-LLM call per
completion — the main model when no auxiliary approval model is configured — and
Desktop's websocket reconnect loop re-runs completion on every reconnect, so an
idle Desktop makes model calls around the clock (#115478).

The listing is also a read-only peek: with no live environment for the session it
must return nothing rather than let ``terminal_tool`` cold-start a sandbox —
Desktop re-fires completion on every reconnect, so an idle chat on a docker
profile piled up foreign ``sleep infinity`` containers (#131751).
"""

import json

import pytest

from tui_gateway.methods_complete import _backend_dir_entries


def _fake_terminal_tool(seen, output="src/\nREADME.md\n", exit_code=0):
    def _capture(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return json.dumps({"output": output, "exit_code": exit_code})

    return _capture


def _live_env(monkeypatch):
    """Simulate an already-running sandbox so the listing path is reachable."""
    monkeypatch.setattr("tools.terminal_tool_lifecycle.get_active_env", lambda task_id: object())




def test_backend_dir_entries_preconfirms_internal_listing(monkeypatch):
    """The listing is Hermes-owned plumbing: it must skip the approval gate, not consult it."""
    _live_env(monkeypatch)
    seen = {}
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_terminal_tool(seen))
    entries = _backend_dir_entries("/workspace", session_key="sess")

    assert seen["kwargs"]["force"] is True
    assert entries == [("README.md", False), ("src", True)]


def test_backend_dir_entries_keeps_session_routing(monkeypatch):
    _live_env(monkeypatch)
    seen = {}
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_terminal_tool(seen))
    _backend_dir_entries("~/proj", session_key="sess-42")

    assert seen["kwargs"]["task_id"] == "sess-42"


@pytest.mark.parametrize("payload", [
    {"output": "boom", "exit_code": 1},
    {"error": "backend unreachable"},
])
def test_backend_dir_entries_swallows_failed_listings(monkeypatch, payload):
    _live_env(monkeypatch)

    def _raw(command, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _raw)
    assert _backend_dir_entries("/workspace", session_key="sess") == []


def test_backend_dir_entries_never_cold_starts_a_sandbox(monkeypatch):
    """No live environment for the session → no listing, and ``terminal_tool`` is never reached:
    creating one here would spawn a foreign container per Desktop reconnect (#131751)."""
    monkeypatch.setattr("tools.terminal_tool_lifecycle.get_active_env", lambda task_id: None)
    seen = {}
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_terminal_tool(seen))

    assert _backend_dir_entries("/workspace", session_key="sess") == []
    assert seen == {}


def test_backend_dir_entries_never_cold_starts_without_a_session(monkeypatch):
    """A session-less completion has no environment of its own to peek at (#131751)."""
    asked = []

    def _probe(task_id):
        asked.append(task_id)
        return None

    monkeypatch.setattr("tools.terminal_tool_lifecycle.get_active_env", _probe)
    seen = {}
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_terminal_tool(seen))

    assert _backend_dir_entries("/workspace", session_key=None) == []
    assert asked == [None]
    assert seen == {}