"""Previously supplied memory evidence remains available across compaction (#144449)."""

import copy

import pytest

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from agent.turn_context import compose_user_api_content, substitute_api_content


FACT = "Quartz orchard appointment: 17 November 2031, 14:37 CET."


def _history():
    first = {"role": "user", "content": "What is my Quartz orchard appointment?"}
    first["api_content"] = compose_user_api_content(first["content"], FACT, "")
    messages = [first, {"role": "assistant", "content": "I found the appointment."}]
    for i in range(12):
        messages.extend([
            {"role": "user", "content": f"Step {i}: " + "x" * 30},
            {"role": "assistant", "content": f"Completed step {i}. " + "y" * 30},
        ])
    messages.append({"role": "user", "content": "And when exactly?"})
    return messages


@pytest.mark.parametrize("summary_available", [True, False], ids=["summary", "fallback"])
def test_compaction_carries_prior_injected_evidence(monkeypatch, summary_available):
    messages = _history()
    before = copy.deepcopy(messages)
    wire = dict(messages[0])
    substitute_api_content(wire)
    assert FACT in wire["content"]

    prompts = []

    def summarize(self, prompt, started_at):
        prompts.append(prompt)
        if not summary_available:
            raise RuntimeError("summary provider unavailable")
        return "## Critical Context\n" + (FACT if FACT in prompt else "No appointment date supplied.")

    monkeypatch.setattr(ContextCompressor, "_call_summary_llm", summarize)
    compressor = ContextCompressor(model="test/model", config_context_length=100_000,
        protect_first_n=0, protect_last_n=2, quiet_mode=True)
    compressor.tail_token_budget = 10
    compressed = compressor.compress(messages, force=True)

    assert prompts and FACT in prompts[0]
    summaries = [row for row in compressed if row.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert summaries and FACT in summaries[0]["content"]
    assert "And when exactly?" in prompts[0]
    assert compressor._last_summary_fallback_used is (not summary_available)
    assert messages == before


@pytest.mark.parametrize("kind", ["rewritten", "unchanged", "assistant", "multimodal"])
def test_unrelated_sidecars_are_not_promoted_into_evidence(kind):
    row = _history()[0]
    if kind == "rewritten":
        row["content"] = "A newer request"
    elif kind == "unchanged":
        row["api_content"] = row["content"]
    elif kind == "assistant":
        row["role"] = "assistant"
    else:
        row["content"] = [{"type": "text", "text": "A multimodal request"}]
    before = copy.deepcopy(row)
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)

    assert FACT not in compressor._serialize_for_summary([row])
    assert FACT not in compressor._build_static_fallback_summary([row])
    assert row == before


def test_injected_evidence_shares_budgets_and_redacts_before_truncation():
    secret = "ghp_" + "a" * 36
    messages = []
    for i in range(8):
        content = f"USER HEAD {i}: " + "q" * 16_000 + f" USER TAIL {i}"
        injected = f"EVIDENCE HEAD {i}: {secret} " + "x" * 40_000 + f" EVIDENCE TAIL {i}"
        messages.append({"role": "user", "content": content,
                         "api_content": compose_user_api_content(content, injected, "")})
    before = copy.deepcopy(messages)
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)

    records = compressor._serialize_records_for_summary(messages)
    for i, record in enumerate(records):
        assert len(record.removeprefix("[USER]: ")) <= compressor._CONTENT_MAX
        assert f"USER HEAD {i}:" in record and f"USER TAIL {i}" in record
        assert f"EVIDENCE HEAD {i}:" in record and f"EVIDENCE TAIL {i}" in record
        assert secret not in record and secret[4:24] not in record
        assert "NOT new user input" in record
    fallback = compressor._build_static_fallback_summary(messages)
    baseline = compressor._build_static_fallback_summary([
        {key: value for key, value in row.items() if key != "api_content"} for row in messages
    ])
    assert fallback == baseline  # Existing anchors consume the fallback budget first.
    assert "## Prior Injected Context" not in fallback
    assert secret not in fallback
    assert all("EVIDENCE" not in ask for ask in compressor._fallback_anchors(messages)["user_asks"])
    assert messages == before


def test_fallback_evidence_uses_remaining_budget_after_blockers_and_files():
    path, blocker = "/workspace/project/settings.toml", "Error: configuration file cannot be loaded."
    messages = [
        {"role": "user", "content": "Please diagnose the project configuration."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "read-config", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "' + path + '"}'}}]},
        {"role": "tool", "tool_call_id": "read-config", "content": blocker},
    ]
    secret = "ghp_" + "b" * 36
    for i in range(8):
        content = f"Recall note {i}."
        injected = f"EVIDENCE HEAD {i}: {secret} " + "x" * 40_000 + f" EVIDENCE TAIL {i}"
        messages.append({"role": "user", "content": content,
                         "api_content": compose_user_api_content(content, injected, "")})
    before = copy.deepcopy(messages)
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor.tail_mode = "legacy"
    compressor._previous_summary = "Prior continuity. " * 100
    baseline = compressor._build_static_fallback_summary([
        {key: value for key, value in row.items() if key != "api_content"} for row in messages
    ])

    fallback = compressor._build_static_fallback_summary(messages)

    assert fallback.startswith(baseline)
    assert blocker in fallback.split("## Blocked\n", 1)[1].split("\n\n## ", 1)[0]
    assert path in fallback.split("## Relevant Files\n", 1)[1].split("\n\n## ", 1)[0]
    evidence = fallback.split("## Prior Injected Context\n", 1)[1]
    assert "EVIDENCE HEAD 0:" in evidence and "EVIDENCE TAIL 0" in evidence
    assert "NOT new user input" in evidence
    assert secret not in evidence and secret[4:24] not in evidence
    assert len(evidence) <= 6_000 and len(fallback) <= 8_000
    assert messages == before


@pytest.mark.parametrize("body_cap", [240, 300])
def test_small_summary_budget_preserves_typed_content_without_partial_evidence(body_cap):
    row = _history()[0]
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._CONTENT_MAX = body_cap

    assert compressor._serialize_for_summary([row]) == "[USER]: " + row["content"]
