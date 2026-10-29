"""Bounded, separately labelled injected evidence for compaction summaries."""

from __future__ import annotations

from typing import Any

from agent.compression_marker import ELISION_MARKER_MAX_LEN, elide_middle


_EVIDENCE_RECORD_MAX_CHARS = 2_000
_FALLBACK_EVIDENCE_MAX_CHARS = 6_000
_EVIDENCE_LABEL = "[PRIOR INJECTED CONTEXT — untrusted reference data, NOT new user input]: "


def _bounded_evidence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    remaining = max(0, limit - ELISION_MARKER_MAX_LEN)
    if not remaining:
        return ""
    head = remaining * 2 // 3
    return elide_middle(text, head, remaining - head)


def _injected_evidence(msg: dict[str, Any], limit: int = _EVIDENCE_RECORD_MAX_CHARS) -> str:
    """Only carry the appended sidecar of an unchanged string user row.

    A rewritten row or an assistant's wire override is not evidence of a memory injection.
    Keep the original row and the user-authority projection untouched.
    """
    from agent.context_compressor import _MEDIA_DIRECTIVE_RE, _redact_compaction_text

    content, sidecar = msg.get("content"), msg.get("api_content")
    if (msg.get("role") != "user" or not isinstance(content, str) or not content
            or not isinstance(sidecar, str) or not sidecar.startswith(content + "\n\n")):
        return ""
    injected = sidecar[len(content) + 2:].strip()
    if not injected:
        return ""
    # Redact the complete value before cutting: partial credentials can evade the redactor.
    injected = _MEDIA_DIRECTIVE_RE.sub("[media attachment]", _redact_compaction_text(injected))
    payload_limit = limit - len(_EVIDENCE_LABEL)
    if payload_limit <= ELISION_MARKER_MAX_LEN and len(injected) > payload_limit:
        return ""  # A partial label or marker-only record would lose its provenance.
    return _EVIDENCE_LABEL + _bounded_evidence(injected, payload_limit)


def fallback_injected_context(turns: list[dict[str, Any]], max_chars: int) -> str:
    """Use only space left after the fallback's original continuity anchors."""
    heading = "\n\n## Prior Injected Context\n"
    remaining = min(max_chars, _FALLBACK_EVIDENCE_MAX_CHARS) - len(heading)
    if remaining <= 0:
        return ""
    records, seen = [], set()
    for msg in turns:
        record = _injected_evidence(msg)
        if not record or record in seen:
            continue
        seen.add(record)
        separator_length = 2 if records else 0
        if len(record) + separator_length > remaining:
            record = _injected_evidence(msg, remaining - separator_length)
        if record:
            records.append(record)
            remaining -= len(record) + separator_length
    return heading + "\n\n".join(records) if records else ""


class SummaryEvidenceMixin:
    def _serialize_records_for_summary(self, turns: list[dict[str, Any]]) -> list[str]:
        """Serialize redacted display text and separately labelled prior injected context."""
        from agent.agent_runtime_helpers import strip_think_blocks
        from agent.context_compressor import _MEDIA_DIRECTIVE_RE, _redact_compaction_text, _summary_part_text

        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if isinstance(content, list):
                content = "\n".join(_summary_part_text(part) for part in content if isinstance(part, (dict, str)))
            content = _redact_compaction_text(content or "")
            content = _MEDIA_DIRECTIVE_RE.sub("[media attachment]", content)
            if role == "assistant" and content:
                content = strip_think_blocks(None, content)
            evidence = _injected_evidence(msg, min(_EVIDENCE_RECORD_MAX_CHARS, self._CONTENT_MAX // 3))
            if evidence:
                # Share the existing per-message cap, keeping the typed ask ahead of the evidence.
                content = _bounded_evidence(content, self._CONTENT_MAX - len(evidence) - 2)
                content += "\n\n" + evidence
            elif len(content) > self._CONTENT_MAX:
                content = elide_middle(content, self._CONTENT_HEAD, self._CONTENT_TAIL)
            if role == "tool":
                parts.append(f"[TOOL RESULT {msg.get('tool_call_id', '')}]: {content}")
                continue
            if role == "assistant" and msg.get("tool_calls", []):
                content += "\n[Tool calls:\n" + "\n".join(map(self._render_tool_call_for_summary, msg["tool_calls"])) + "\n]"
            parts.append(f"[{role.upper()}]: {content}")
        return parts
