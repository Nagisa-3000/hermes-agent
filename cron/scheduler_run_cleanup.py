"""Release a cron caller immediately while retaining resources for an unfinished worker."""

from __future__ import annotations


def _finish_inline(scope, session_db, agent, job_id, job_name, session_id, deferred_agents):
    from cron.scheduler import _finalize_cron_session, _teardown_cron_agent

    try:
        if session_db:
            _finalize_cron_session(session_db, agent, job_id, job_name, session_id,
                                   workdir=scope.workdir)
    finally:
        try:
            if deferred_agents is not None:
                # Delivery needs a live async client (#58720); the turn itself is done.
                if agent is not None:
                    deferred_agents.append(agent)
            else:
                _teardown_cron_agent(agent, job_id)
        finally:
            scope.release()


def finish_run(scope, future, session_db, agent, job_id, job_name, session_id, deferred_agents):
    from cron.scheduler_detached_worker import defer_teardown_to_running_worker

    deferred = False
    try:
        deferred = defer_teardown_to_running_worker(
            future, session_db, agent, job_id, job_name, session_id,
            workdir=scope.workdir, on_finish=scope.release)
    finally:
        try:
            # ContextVar tokens belong to this caller, never the Future callback's context.
            scope.exit()
        finally:
            if not deferred:
                _finish_inline(scope, session_db, agent, job_id, job_name, session_id, deferred_agents)
