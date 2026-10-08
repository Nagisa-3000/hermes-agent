"""Cron terminal ownership follows the worker lifetime, including detached workers."""
import json
from concurrent.futures import Future
from contextvars import copy_context
from pathlib import Path
from threading import Thread

import pytest

from cron import scheduler
from cron.scheduler_run_scope import _CronRunScope
from tools import terminal_tool
from tools.terminal_tool_lifecycle import cleanup_vm
from tui_gateway import server as gateway


def _homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    work = default / "profiles" / "work"
    for home in (default, work):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(json.dumps({
            "model": "test/model", "cron": {"preflight": False},
            "terminal": {"backend": "local"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default))
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    return default, work


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("profile", ["default", "work"])
def test_run_job_retains_owner_until_worker_and_cleanup_finish(tmp_path, monkeypatch, pending, profile):
    default, work = _homes(tmp_path, monkeypatch)
    home = default if profile == "default" else work
    observed = {}
    future = Future()
    future.set_running_or_notify_cancel()

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def close(self):
            observed["close_key"] = terminal_tool._resolve_container_task_id(observed["task_id"])
            cleanup_vm(observed["key"])

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "api_key": "test-key", "base_url": "https://example.invalid/v1",
        "provider": "openrouter", "api_mode": "chat_completions"})

    def controlled_worker(agent, prompt, job, job_id, job_name, task_id, cancel_event, *, worker_state):
        observed["task_id"] = task_id
        observed["context"] = copy_context()
        observed["key"] = terminal_tool._resolve_container_task_id(task_id)
        seeded = json.loads(terminal_tool.terminal_tool("export CRON_LIFETIME_PROBE=own-run", task_id=task_id))
        assert seeded["exit_code"] == 0
        worker_state["future"] = future
        if not pending:
            future.set_result({"final_response": "done"})
        return {"final_response": "done"}

    monkeypatch.setattr(scheduler, "_run_agent_with_watchdog", controlled_worker)
    try:
        with gateway._session_profile_runtime_scope({"profile_home": str(home)}, hydrate_secrets=False):
            assert scheduler.run_job({"id": "lifetime-job", "prompt": "hello"}, execution_id="same-run")[0]
            context = observed["context"]
            if pending:
                assert context.run(terminal_tool._resolve_container_task_id, observed["task_id"]) == observed["key"]
                late = json.loads(context.run(terminal_tool.terminal_tool,
                    'printf %s "$CRON_LIFETIME_PROBE"', task_id=observed["task_id"]))
                assert late["output"] == "own-run"
        if pending:
            # The completing thread has no profile bindings or caller-created ContextVar tokens.
            completion = Thread(target=future.set_result, args=({"final_response": "late"},))
            completion.start()
            completion.join(timeout=10)
            assert not completion.is_alive()
        assert observed["close_key"] == observed["key"]
        assert context.run(terminal_tool._resolve_container_task_id, observed["task_id"]) != observed["key"]
    finally:
        if not future.done():
            future.set_result({"final_response": "late"})
        if "key" in observed:
            cleanup_vm(observed["key"])
            cleanup_vm("default")


def test_same_run_id_in_two_profiles_has_independent_ownership(tmp_path, monkeypatch):
    default, work = _homes(tmp_path, monkeypatch)
    runs = []
    for home in (default, work):
        with gateway._session_profile_runtime_scope({"profile_home": str(home)}, hydrate_secrets=False):
            context = copy_context()
            scope = context.run(_CronRunScope, {"id": "same-job"}, "same-job", "same-run")
            runs.append((context, scope))
    try:
        keys = [context.run(terminal_tool._resolve_container_task_id, scope.task_id) for context, scope in runs]
        assert keys[0] != keys[1]
        first_context, first = runs[0]
        first_context.run(first.release)
        second_context, second = runs[1]
        assert second_context.run(terminal_tool._resolve_container_task_id, second.task_id) == keys[1]
    finally:
        for context, scope in runs:
            context.run(scope.exit)
            context.run(scope.release)
