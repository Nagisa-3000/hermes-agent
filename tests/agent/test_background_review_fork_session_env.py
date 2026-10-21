"""Regression coverage for #142720: the background review fork must not republish its own
session id into the parent process's environment.

``AIAgent.__init__`` mints a session id and publishes it through
``agent_init._publish_session_id`` -> ``gateway.session_context.set_current_session_id``, which
writes ``os.environ["HERMES_SESSION_ID"]`` for every caller that is not a delegated child. The
review fork is built in the parent process and never persists that id (``_persist_disabled``,
``_session_db = None``), so the write left the parent's environment — and every tool subprocess
that inherits it — pointed at a session with no row in ``state.db``, while the fork itself got the
parent's id back on the attribute only.
"""

import os
from unittest.mock import patch

import pytest


def _make_agent_stub(agent_cls):
    """Minimal parent agent: enough state for ``build_cache_parity_fork`` to run."""
    import datetime as _dt

    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "parent-session-abc"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = "PARENT-SYSTEM-PROMPT"
    agent.ephemeral_system_prompt = None
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.enabled_toolsets = ["memory", "skills"]
    agent.disabled_toolsets = None
    agent.reasoning_config = {"enabled": True, "effort": "medium"}
    agent.prefill_messages = None
    agent.providers_allowed = None
    agent.providers_ignored = None
    agent.providers_order = None
    agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    return agent


def _make_publishing_recorder():
    """Stand-in for the fork's ``AIAgent`` that publishes its session id like the real one.

    The real constructor reaches ``_publish_session_id`` from ``_init_session_state``; calling it
    here keeps the test on the production publication path instead of reproducing its rules.
    """

    class _Recorder:
        def __init__(self, *args, **kwargs):
            from agent.agent_init import _publish_session_id

            self.session_id = "review-fork-phantom-id"
            self.tools = []
            self.valid_tool_names = set()
            self._tool_snapshot_generation = 0
            self._cached_system_prompt = None
            self.suppress_status_output = False
            self._memory_write_origin = None
            self._memory_write_context = None
            self._memory_store = None
            self._memory_enabled = None
            self._user_profile_enabled = None
            self._memory_nudge_interval = None
            self._skill_nudge_interval = None
            self.session_start = None
            self.ephemeral_system_prompt = None
            _publish_session_id(self.session_id)

        def close(self):
            pass

        def shutdown_memory_provider(self):
            pass

    return _Recorder


@pytest.fixture(autouse=True)
def _clean_session_env():
    previous = os.environ.get("HERMES_SESSION_ID")
    yield
    if previous is None:
        os.environ.pop("HERMES_SESSION_ID", None)
    else:
        os.environ["HERMES_SESSION_ID"] = previous


def test_fork_construction_leaves_the_parent_session_id_in_the_env():
    import run_agent
    from agent.background_review import build_cache_parity_fork

    agent = _make_agent_stub(run_agent.AIAgent)
    os.environ["HERMES_SESSION_ID"] = agent.session_id

    _Recorder = _make_publishing_recorder()
    with patch.object(run_agent, "AIAgent", _Recorder):
        fork, _rt, _routed = build_cache_parity_fork(agent, max_iterations=5)

    # The fork still carries the parent's id on the instance (prefix-cache warmth).
    assert fork.session_id == agent.session_id
    # The phantom id never reaches the process environment...
    assert os.environ["HERMES_SESSION_ID"] == agent.session_id
    # ...nor the ContextVar that tools read first.
    from gateway.session_context import get_session_env

    assert get_session_env("HERMES_SESSION_ID") == agent.session_id


def test_fork_construction_does_not_leave_a_phantom_id_for_a_later_reader():
    """A reader outside the construction scope resolves the parent, not the fork (#142720)."""
    import run_agent
    from agent.background_review import build_cache_parity_fork

    agent = _make_agent_stub(run_agent.AIAgent)
    os.environ.pop("HERMES_SESSION_ID", None)

    _Recorder = _make_publishing_recorder()
    with patch.object(run_agent, "AIAgent", _Recorder):
        build_cache_parity_fork(agent, max_iterations=5)

    # No env entry means no tool subprocess inherits a phantom id; if one exists it must be the
    # parent's, never the fork's.
    assert os.environ.get("HERMES_SESSION_ID") in (None, agent.session_id)
