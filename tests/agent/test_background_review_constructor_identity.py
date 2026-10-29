"""Real-constructor session identity contracts for detached forks (#142720)."""

import os
import subprocess
import sys
import threading

import pytest

from agent.background_review import build_cache_parity_fork
from gateway.session_context import (
    get_session_env,
    scoped_current_session_id,
    set_current_session_id,
)
from hermes_constants import get_hermes_home
from run_agent import AIAgent
from tools.environments.local import build_subprocess_env


class ConstructionStopped(RuntimeError):
    """Stop after the real constructor has published its freshly minted id."""


@pytest.fixture
def parent_agent(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "parent-live-session")
    # The identity contract needs a constructor, without probing a model endpoint.
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  default: test-model\n  provider: openai\n"
        "  base_url: http://127.0.0.1:9/v1\n  context_length: 131072\n",
        encoding="utf-8",
    )
    with scoped_current_session_id("parent-live-session"):
        parent = AIAgent(
            model="test-model",
            provider="openai",
            api_key="test-key",
            base_url="http://127.0.0.1:9/v1",
            session_id="parent-live-session",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        parent._cached_system_prompt = "stable-parent-prompt"
        try:
            yield parent
        finally:
            parent.close()


def _subprocess_session_id():
    return subprocess.check_output(
        [sys.executable, "-c", 'import os; print(os.environ.get("HERMES_SESSION_ID", ""))'],
        env=build_subprocess_env(),
        text=True,
        timeout=20,
    ).strip()


@pytest.mark.parametrize("write_origin", ["background_review", "side_question"])
@pytest.mark.parametrize("constructor_raises", [False, True])
def test_real_fork_constructor_preserves_caller_identity(
    parent_agent, monkeypatch, write_origin, constructor_raises,
):
    import run_agent

    constructed = []

    def construct(**kwargs):
        fork = AIAgent(**kwargs)
        constructed.append(fork)
        if constructor_raises:
            raise ConstructionStopped
        return fork

    monkeypatch.setattr(run_agent, "AIAgent", construct)
    try:
        if constructor_raises:
            with pytest.raises(ConstructionStopped):
                build_cache_parity_fork(parent_agent, max_iterations=1, write_origin=write_origin)
        else:
            fork, _, routed = build_cache_parity_fork(
                parent_agent, max_iterations=1, write_origin=write_origin,
            )
            assert not routed
            assert fork.session_id == parent_agent.session_id
            assert fork._cached_system_prompt == parent_agent._cached_system_prompt

        assert os.environ["HERMES_SESSION_ID"] == parent_agent.session_id
        assert get_session_env("HERMES_SESSION_ID") == parent_agent.session_id
        assert _subprocess_session_id() == parent_agent.session_id
    finally:
        for fork in constructed:
            fork.close()


@pytest.mark.parametrize("write_origin", ["background_review", "side_question"])
@pytest.mark.parametrize("constructor_raises", [False, True])
def test_fork_construction_does_not_roll_back_a_concurrent_live_session(
    parent_agent, monkeypatch, write_origin, constructor_raises,
):
    import run_agent

    constructed = threading.Event()
    release = threading.Event()
    result = {}
    forks = []

    def construct(**kwargs):
        fork = AIAgent(**kwargs)
        forks.append(fork)
        constructed.set()
        assert release.wait(30), "live session did not release the constructor"
        if constructor_raises:
            raise ConstructionStopped
        return fork

    def build():
        with scoped_current_session_id(parent_agent.session_id):
            try:
                build_cache_parity_fork(parent_agent, max_iterations=1, write_origin=write_origin)
            except ConstructionStopped:
                result["constructor_raised"] = True
            except Exception as exc:
                result["error"] = exc
            finally:
                result["context"] = get_session_env("HERMES_SESSION_ID")
                result["subprocess"] = _subprocess_session_id()

    monkeypatch.setattr(run_agent, "AIAgent", construct)
    worker = threading.Thread(target=build, daemon=True)
    worker.start()
    try:
        assert constructed.wait(30), "real fork constructor did not finish"
        during_construction = os.environ["HERMES_SESSION_ID"]
        set_current_session_id("other-live-session")
    finally:
        release.set()
        worker.join(30)
        for fork in forks:
            fork.close()

    assert not worker.is_alive()
    assert "error" not in result
    assert bool(result.get("constructor_raised")) == constructor_raises
    assert during_construction == parent_agent.session_id
    assert os.environ["HERMES_SESSION_ID"] == "other-live-session"
    assert get_session_env("HERMES_SESSION_ID") == "other-live-session"
    assert result["context"] == parent_agent.session_id
    assert result["subprocess"] == parent_agent.session_id
    assert _subprocess_session_id() == "other-live-session"
