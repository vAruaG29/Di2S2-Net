"""
In-memory job registry for Mode-2 inference runs.

A job has a uuid, a status (queued | running | done | failed), and a
list of event dicts emitted by the pipeline runner (parsed from
StepTimer stdout). Events are appended live and broadcast via an
asyncio.Queue so the SSE endpoint can stream them.

Restart wipes the registry — intentional, per the plan.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


@dataclass
class Job:
    id: str
    dataset_name: str
    source: str  # "existing" | "upload"
    status: JobStatus = "queued"
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    # The subprocess currently running this job's phase (if any). The
    # cancel endpoint reaches in here to kill it.
    current_proc: asyncio.subprocess.Process | None = None
    # Set when the user has requested cancellation; the runner checks this
    # between phases too.
    cancel_requested: bool = False
    # one async queue per subscriber (the SSE endpoint).
    _subscribers: list[asyncio.Queue] = field(default_factory=list)

    def add_event(self, ev: dict) -> None:
        self.events.append(ev)
        for q in self._subscribers:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        # replay everything we have so far so a late subscriber catches up.
        for ev in self.events:
            q.put_nowait(ev)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def request_cancel(self) -> bool:
        """
        Mark the job for cancellation and try to terminate the running
        subprocess if any. Returns True if a process was signalled.
        """
        self.cancel_requested = True
        proc = self.current_proc
        if proc is None or proc.returncode is not None:
            return False
        try:
            proc.terminate()
        except ProcessLookupError:
            return False
        return True


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def create(self, dataset_name: str, source: str) -> Job:
        j = Job(id=uuid.uuid4().hex, dataset_name=dataset_name, source=source)
        self._jobs[j.id] = j
        return j

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())


REGISTRY = JobRegistry()
