"""The job queue: one worker thread runs every job, so GPU models never run side by side.

Heavy backends (big weights) are loaded on demand; switching to another heavy backend
unloads the previous one first, and all of them are unloaded after ``idle_unload_s``
without jobs. Unloading is done by the worker between jobs only.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from . import memory
from .backends.base import Backend, Registry
from .config import Settings
from .errors import BackendUnavailable, JobCancelled, friendly, is_cuda_oom, is_ram_oom

log = logging.getLogger("dronegen.jobs")

QUEUED, RUNNING, DONE, ERROR, CANCELLED = "queued", "running", "done", "error", "cancelled"
FINISHED = frozenset({DONE, ERROR, CANCELLED})


@dataclass
class Job:
    id: str
    kind: str
    backend: str
    params: dict
    inputs: dict = field(default_factory=dict, repr=False)  # in-memory inputs (an uploaded image)
    status: str = QUEUED
    progress: float = 0.0
    stage: str = "в очереди"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # role -> file name served by /files
    result: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self, queue_position: int | None = None) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "backend": self.backend,
            "params": self.params,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "error": self.error,
            "warnings": list(self.warnings),
            "files": dict(self.files),
            "result": self.result,
            "metrics": self.metrics,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "queue_position": queue_position,
        }


class JobContext:
    """What a backend sees while running a job: progress, warnings, cancellation, output files."""

    def __init__(self, job: Job, manager: "JobManager"):
        self.job = job
        self.manager = manager
        self.settings: Settings = manager.settings

    def progress(self, fraction: float, stage: str | None = None) -> None:
        self.check_cancelled()
        self.job.progress = max(self.job.progress, min(max(float(fraction), 0.0), 0.999))
        if stage:
            self.job.stage = stage

    def warn(self, message: str) -> None:
        log.warning("%s: %s", self.job.id, message)
        if message not in self.job.warnings:
            self.job.warnings.append(message)

    def check_cancelled(self) -> None:
        if self.job.cancel_event.is_set():
            raise JobCancelled()

    @property
    def cancelled(self) -> bool:
        return self.job.cancel_event.is_set()

    def output(self, role: str, ext: str) -> Path:
        name = f"{self.job.id}_{role}.{ext}"
        self.job.files[role] = name
        return self.settings.outputs / name


class JobManager:
    def __init__(self, settings: Settings, registry: Registry):
        self.settings = settings
        self.registry = registry
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._unload_requested = threading.Event()
        self._worker: threading.Thread | None = None
        self._running: Job | None = None
        self._last_activity = time.time()
        # called in the worker before a GPU job: other GPU users (the local LLM) step aside
        self.gpu_hooks: list = []

    # -- lifecycle ----------------------------------------------------------------------
    def start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = threading.Thread(target=self._loop, name="dronegen-worker", daemon=True)
            self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._running is not None:
            self._running.cancel_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        self.unload_all()

    # -- API ----------------------------------------------------------------------------
    def submit(self, kind: str, backend_name: str, params: dict, inputs: dict | None = None) -> Job:
        backend = self.registry.get(kind, backend_name)
        ok, reason = backend.check()
        if not ok:
            raise BackendUnavailable(f"{backend.title}: {reason}")
        params = backend.validate(params)
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, backend=backend_name, params=params, inputs=inputs or {})
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        self._queue.put(job.id)
        log.info("job %s queued: %s/%s", job.id, kind, backend_name)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def queue_position(self, job: Job) -> int | None:
        if job.status != QUEUED:
            return None
        with self._lock:
            ahead = [j for j in self._jobs.values() if j.status == QUEUED and j.created < job.created]
        return len(ahead) + (1 if self._running is not None else 0)

    def pending(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == QUEUED)

    @property
    def running(self) -> Job | None:
        return self._running

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.status in FINISHED:
            return job
        job.cancel_event.set()
        if job.status == QUEUED:
            self._finish(job, CANCELLED, "отменено")
        return job

    def request_unload(self) -> None:
        """Free GPU/RAM held by loaded models (done by the worker as soon as it is idle)."""
        self._unload_requested.set()

    def release_gpu(self, timeout: float = 900.0) -> bool:
        """Wait until the queue is idle, then unload the GPU models (for the local LLM).

        Blocking: call it from a thread. False if the queue was still busy after ``timeout``.
        """
        t0 = time.time()
        while self._running is not None or self.pending():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.2)
        if not self._loaded():
            return True
        self.request_unload()
        while self._loaded():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.1)
        return True

    # -- worker -------------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._idle()
                continue
            job = self.get(job_id)
            if job is None or job.status != QUEUED:
                continue
            self._run(job)

    def _idle(self) -> None:
        idle_for = time.time() - self._last_activity
        timeout = self.settings.idle_unload_s
        if self._unload_requested.is_set() or (timeout and idle_for > timeout and self._loaded()):
            self._unload_requested.clear()
            if self._loaded():
                log.info("unloading models (%s)", "request" if idle_for <= timeout else "idle")
            self.unload_all()

    def _loaded(self) -> list[Backend]:
        return [b for b in self.registry.all() if b.loaded and b.heavy]

    def unload_all(self) -> None:
        for b in self.registry.all():
            if b.loaded:
                try:
                    b.unload()
                except Exception:  # noqa: BLE001
                    log.exception("unload of %s failed", b.name)
                b.loaded = False
        memory.free_gpu()

    def _ensure_loaded(self, backend: Backend, ctx: JobContext) -> None:
        if backend.loaded:
            return
        if backend.heavy:
            for other in self._loaded():
                if other is not backend:
                    log.info("unloading %s to make room for %s", other.name, backend.name)
                    other.unload()
                    other.loaded = False
            memory.free_gpu()
        ctx.progress(0.0, "загрузка модели")
        t0 = time.perf_counter()
        backend.load(ctx)
        backend.loaded = True
        ctx.job.metrics["load_seconds"] = round(time.perf_counter() - t0, 2)

    def _run(self, job: Job) -> None:
        backend = self.registry.get(job.kind, job.backend)
        ctx = JobContext(job, self)
        self._running = job
        job.status, job.started, job.stage = RUNNING, time.time(), "подготовка"
        log.info("job %s started", job.id)
        try:
            if backend.needs_gpu:
                for hook in self.gpu_hooks:
                    hook()
            self._ensure_loaded(backend, ctx)
            with memory.PeakMonitor() as peak:
                job.result = backend.run(ctx, job.params, job.inputs) or {}
            job.metrics.update(peak.metrics())
            self._finish(job, DONE, "готово")
        except JobCancelled:
            self._finish(job, CANCELLED, "отменено")
        except Exception as e:  # noqa: BLE001 - every failure becomes a readable job error
            log.exception("job %s failed", job.id)
            job.error = friendly(e)
            if backend.heavy and (is_cuda_oom(e) or is_ram_oom(e)):
                backend.unload()
                backend.loaded = False
            self._finish(job, ERROR, "ошибка")
        finally:
            job.inputs.clear()
            self._running = None
            self._last_activity = time.time()
            if backend.needs_gpu:
                memory.free_gpu()

    def _finish(self, job: Job, status: str, stage: str) -> None:
        job.status, job.stage, job.finished = status, stage, time.time()
        if status == DONE:
            job.progress = 1.0
        if job.started:
            job.metrics.setdefault("total_seconds", round(job.finished - job.started, 2))
        log.info("job %s %s%s", job.id, status, f": {job.error}" if job.error else "")

    def _prune(self) -> None:
        extra = len(self._jobs) - self.settings.max_jobs_kept
        for job_id in list(self._jobs)[: max(0, extra)]:
            if self._jobs[job_id].status in FINISHED:
                del self._jobs[job_id]
