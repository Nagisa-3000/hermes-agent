"""Durable completion rows that are not model turns."""

from __future__ import annotations

import time
from typing import Any, Optional

from hermes_state_messages import (
    _INSERT_MESSAGE_SQL, _coerce_timestamp, stamp_message_uid,
)


class SessionDeliveriesMixin:
    """Persist idempotent background delivery notices between user turns."""

    def append_delegation_delivery(self, session_id: str, content: str, metadata: dict[str, Any]) -> int:
        """Record a detached API result once, including replay after compression rotation."""
        delegation_id = metadata.get("delegation_id")
        if not delegation_id:
            raise ValueError("Delegation delivery requires a stable delegation_id")
        msg = {
            "content": content,
            "display_kind": "hidden" if metadata.get("presentation_suppressed") else "async_delegation_complete",
            "display_metadata": metadata,
        }
        stamp_message_uid(msg)
        params = self._message_row_params(session_id, "user", msg, None, _coerce_timestamp(None, time.time()), keep_reasoning=True)

        def _do(conn):
            existing = conn.execute(
                """WITH RECURSIVE lineage(id) AS (
                    SELECT ? UNION
                    SELECT s.parent_session_id FROM sessions s JOIN lineage l ON s.id = l.id
                    JOIN sessions p ON p.id = s.parent_session_id WHERE p.end_reason = 'compression'
                ) SELECT m.id FROM messages m JOIN lineage l ON m.session_id = l.id
                WHERE m.display_kind IN ('async_delegation_complete', 'hidden')
                AND json_extract(m.display_metadata, '$.delegation_id') = ?
                AND coalesce(json_extract(m.display_metadata, '$.delivery_notice'), '') = ? LIMIT 1""",
                (session_id, delegation_id, metadata.get("delivery_notice", ""))).fetchone()
            if existing is not None:
                return existing[0]
            self._check_transcript_write_guards(conn, session_id, None, reject_active_turn_lease=True)
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
            self._bump_session_counters(conn, session_id, 1, 0, unit=True)
            return msg_id

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def append_cron_delivery(
        self, session_id: str, content: str, delivery_id: str, *,
        job_name: Optional[str] = None, timestamp: Any = None,
    ) -> int:
        """Record a cron no_agent result as a display-only timeline row."""
        if not session_id or not delivery_id:
            raise ValueError("Cron delivery requires a session id and delivery id")
        metadata = {"cron_delivery_id": str(delivery_id), "display_text": content}
        if job_name:
            metadata["job_name"] = job_name
        msg = {"content": content, "display_kind": "process_complete", "display_metadata": metadata}
        stamp_message_uid(msg)
        params = self._message_row_params(
            session_id, "user", msg, None, _coerce_timestamp(timestamp, time.time()), keep_reasoning=True)

        def _do(conn):
            existing = conn.execute(
                """WITH RECURSIVE lineage(id) AS (
                    SELECT ? UNION
                    SELECT s.parent_session_id FROM sessions s JOIN lineage l ON s.id = l.id
                    JOIN sessions p ON p.id = s.parent_session_id WHERE p.end_reason = 'compression'
                ) SELECT m.id, m.content, m.display_metadata FROM messages m JOIN lineage l ON m.session_id = l.id
                WHERE m.display_kind = 'process_complete'
                AND json_extract(m.display_metadata, '$.cron_delivery_id') = ? LIMIT 1""",
                (session_id, str(delivery_id))).fetchone()
            if existing is not None:
                existing_content = self._decode_content(existing[1])
                existing_metadata = self._decode_display_metadata(existing[2]) or {}
                if (existing_content != content
                        or existing_metadata.get("display_text") != content):
                    raise ValueError("cron delivery id already belongs to a different payload")
                return existing[0]
            self._check_transcript_write_guards(conn, session_id, None, reject_active_turn_lease=True)
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
            self._bump_session_counters(conn, session_id, 1, 0, unit=True)
            return msg_id

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
