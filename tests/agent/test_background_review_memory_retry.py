"""Automatic memory reviews retain their trigger across foreground preemption (#145920)."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import background_review
from run_agent import AIAgent
from agent.background_review import cancel_background_review_for_live_turn
from tests.waiting import wait_until


def _response(content="ok", *, save=None):
    calls = None
    if save is not None:
        calls = [
            SimpleNamespace(
                id="save-preference",
                type="function",
                function=SimpleNamespace(
                    name="memory",
                    arguments=json.dumps(
                        {
                            "action": "add",
                            "target": "user",
                            "content": save,
                        }
                    ),
                ),
            )
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls),
                finish_reason="tool_calls" if calls else "stop",
            )
        ],
        model="test-model",
        usage=None,
    )


@pytest.fixture
def review_session(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = home / "config.yaml"
    config.write_text(
        "auxiliary:\n  background_review:\n    enabled: true\n", encoding="utf-8"
    )
    client = MagicMock()
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: client)
    parent = AIAgent(
        model="test-model",
        provider="openai",
        api_key="test-key",
        api_mode="chat_completions",
        base_url="https://api.openai.com/v1",
        enabled_toolsets=["memory"],
        quiet_mode=True,
        skip_context_files=True,
        save_trajectories=False,
        max_iterations=4,
    )
    parent.compression_enabled = False
    parent._memory_nudge_interval = 3
    parent._skill_nudge_interval = 0
    state = SimpleNamespace(
        parent=parent,
        client=client,
        home=home,
        config=config,
        started=threading.Event(),
        release=threading.Event(),
        written=threading.Event(),
        review_requests=[],
        block_first=True,
        late_ack=False,
        first_run=None,
    )
    finish = background_review.finish_background_review_run

    def finish_request(agent, run, **kwargs):
        if state.late_ack and run is state.first_run and run.cancel_requested.is_set():
            assert state.release.wait(10), "cancelled request was never released"
        finish(agent, run, **kwargs)

    monkeypatch.setattr(
        background_review, "finish_background_review_run", finish_request
    )

    def complete(**kwargs):
        messages = kwargs["messages"]
        if not any(
            message["role"] == "user"
            and isinstance(message["content"], str)
            and message["content"].startswith(parent._MEMORY_REVIEW_PROMPT)
            for message in messages
        ):
            return _response("Understood.")
        if not state.review_requests:
            state.review_requests.append(messages)
            state.first_run = parent._background_review_run
            state.started.set()
            if state.block_first:
                fork = parent._background_review_agent
                wait_until(
                    lambda: fork._hard_interrupt_requested.is_set(),
                    what="review provider cancellation",
                )
                return _response("Cancelled.")
        elif messages[-1]["role"] == "tool":
            state.written.set()
            return _response("Saved.")
        else:
            state.review_requests.append(messages)
        latest = next(
            message["content"]
            for message in reversed(messages[:-1])
            if message["role"] == "user"
        )
        return _response("", save=latest)

    client.chat.completions.create.side_effect = complete
    yield state
    state.release.set()
    run = parent._background_review_run
    if run is not None:
        cancel_background_review_for_live_turn(parent)
        assert run.request_done.wait(10), "review worker did not finish"
    parent.release_clients()


@pytest.mark.parametrize("mode", ["cooperative", "late-ack", "start-failure"])
def test_preempted_automatic_review_retries_latest_turn_and_writes_memory(
    review_session, monkeypatch, mode
):
    state = review_session
    parent = state.parent
    state.late_ack = mode == "late-ack"
    state.block_first = mode != "start-failure"
    if mode == "start-failure":
        start = threading.Thread.start
        failed = threading.Event()

        def start_review(thread):
            if thread.name == "bg-review" and not failed.is_set():
                failed.set()
                raise RuntimeError("review worker could not start")
            start(thread)

        monkeypatch.setattr(threading.Thread, "start", start_review)
    parent._turns_since_memory = parent._memory_nudge_interval - 1
    assert (
        parent.run_conversation("User prefers tea.")["final_response"] == "Understood."
    )
    if mode != "start-failure":
        assert state.started.wait(10), "automatic review never reached the provider"
    first_run = parent._background_review_run
    frozen_prompt = parent._cached_system_prompt
    assert (
        parent.run_conversation("User prefers coffee.")["final_response"]
        == "Understood."
    )
    latest = "User prefers coffee."
    if state.late_ack:
        state.release.set()
        assert first_run.request_done.wait(10), "preempted review never acknowledged"
        latest = "User prefers cocoa."
        assert parent.run_conversation(latest)["final_response"] == "Understood."
    assert state.written.wait(10), "cancelled automatic memory review was never retried"
    wait_until(lambda: parent._background_review_run is None, what="memory review exit")
    assert len(state.review_requests) == (1 if mode == "start-failure" else 2)
    assert latest in (state.home / "memories" / "USER.md").read_text(encoding="utf-8")
    assert parent._cached_system_prompt == frozen_prompt


@pytest.mark.parametrize(
    "mode", ["completed", "explicit", "focused", "disabled", "session-reset"]
)
def test_reviews_without_automatic_memory_debt_do_not_retry(review_session, mode):
    state = review_session
    parent = state.parent
    parent._turns_since_memory = 0
    state.block_first = mode != "completed"
    if mode == "disabled":
        state.config.write_text(
            "auxiliary:\n  background_review:\n    enabled: false\n",
            encoding="utf-8",
        )
    parent._spawn_background_review(
        [
            {"role": "user", "content": "User prefers tea."},
            {"role": "assistant", "content": "Understood."},
        ],
        review_memory=True,
        explicit=mode == "explicit",
        focus="save the preference" if mode == "focused" else None,
    )
    if mode != "disabled":
        assert state.started.wait(10), "review never reached the provider"
        if mode == "completed":
            assert state.written.wait(10), "completed review never saved memory"
            wait_until(
                lambda: parent._background_review_run is None, what="review exit"
            )
        elif mode == "session-reset":
            parent.reset_session_state()
    assert (
        parent.run_conversation("User prefers coffee.")["final_response"]
        == "Understood."
    )
    assert parent._background_review_run is None
    assert len(state.review_requests) == (0 if mode == "disabled" else 1)
