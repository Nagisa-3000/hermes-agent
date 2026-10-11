"""Transcript-only delivery for no_agent cron output sent to Bot Chat."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cron.scheduler")


_NO_AGENT_BOT_CHAT_BUSY = "busy"


def _record_receipt(job: dict, profile_label: str, delivery_id: str, status: str) -> None:
    target = f"bot-chat:{profile_label}"
    job.setdefault("_bot_chat_delivery_receipts", {})[target] = {
        "status": status, "delivery_id": delivery_id}


def _append(
    job: dict, content: str, home: Path, delivery_id: str,
) -> tuple[str, Optional[str]]:
    from hermes_state import SessionDB
    from hermes_state_errors import CompressionSessionClosedError, SessionTurnLeaseLostError
    from hermes_state_ids import new_session_id
    from cron import scheduler_delivery as delivery

    db = None
    try:
        db = SessionDB(db_path=home / "state.db")
        row = db.get_session_by_title("Bot Chat")
        if row is None:
            session_id = new_session_id()
            db.create_session(session_id, source="cli")
            try:
                db.set_session_title(session_id, "Bot Chat")
            except ValueError:
                row = db.get_session_by_title("Bot Chat")
                if row is None:
                    raise
                session_id = row["id"]
        else:
            session_id = db.get_compression_tip(row["id"]) or row["id"]
        job_name = delivery._redact_cron_payload(
            job.get("name") or job.get("id", "cron"), "job name")
        try:
            db.append_cron_delivery(session_id, content, delivery_id, job_name=job_name)
        except CompressionSessionClosedError:
            session_id = db.get_compression_tip(session_id) or session_id
            db.append_cron_delivery(session_id, content, delivery_id, job_name=job_name)
    except SessionTurnLeaseLostError:
        return _NO_AGENT_BOT_CHAT_BUSY, None
    except Exception as exc:
        logger.warning("Job '%s': no_agent Bot Chat transcript append failed: %s",
                       job.get("id", "?"), exc, exc_info=True)
        return "error", f"bot-chat transcript append failed: {type(exc).__name__}: {exc}"
    finally:
        if db is not None:
            with contextlib.suppress(Exception):
                db.close()
    return "settled", None


def deliver_no_agent_bot_chat(
    job: dict, content: str, profile_label: str, home: Path,
    delivery_id: str, *, deferred: Optional[dict] = None,
) -> Optional[str]:
    """Persist a no_agent result, or enqueue it behind existing Bot Chat work."""
    from cron.bot_chat_delivery import defer, has_pending_for_home, read_pending, requeue
    from tools.bot_live_delivery import has_pending_delivery, read_delivery_result

    target = f"bot-chat:{profile_label}"
    prior_receipt = read_delivery_result(home, delivery_id) if deferred is None else None
    if prior_receipt is not None:
        status = prior_receipt["status"]
        _record_receipt(job, profile_label, delivery_id, status)
        if status == "settled":
            return None
        detail = ("completion unverified; do not resend" if status in ("queued", "claimed")
                  else prior_receipt.get("error") or prior_receipt.get("reason") or "not completed")
        return f"{target} {status} (receipt {delivery_id}): {detail}"
    pending = read_pending(delivery_id) if deferred is None else None
    if pending is not None:
        status = pending["status"]
        _record_receipt(job, profile_label, delivery_id, status)
        if status in ("settled", "suppressed"):
            return None
        return f"{target} {status} (receipt {delivery_id}): completion unverified; do not resend"

    prior_queued = has_pending_for_home(
        home,
        before_sequence=int(deferred["sequence"]) if deferred is not None else None,
        exclude_id=delivery_id,
    )
    if not prior_queued and not has_pending_delivery(home):
        status, error = _append(job, content, home, delivery_id)
        if status == "settled":
            _record_receipt(job, profile_label, delivery_id, status)
            return None
        if status != _NO_AGENT_BOT_CHAT_BUSY:
            return error
        if deferred is not None:
            pending = requeue(delivery_id)
            if pending is None:
                return f"{target} queued (receipt {delivery_id}): completion unverified; do not resend"
            _record_receipt(job, profile_label, delivery_id, pending["status"])
            return f"{target} {pending['status']} (receipt {delivery_id}): completion unverified; do not resend"

    pending = defer(delivery_id, dict(job), content, profile_label, home, for_failure=False)
    status = pending["status"]
    _record_receipt(job, profile_label, delivery_id, status)
    return f"{target} {status} (receipt {delivery_id}): completion unverified; do not resend"
