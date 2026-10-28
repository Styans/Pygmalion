"""Background work that never freezes Blender.

Threads talk to the server and never touch ``bpy``: they post events to a queue.
``tick()`` runs on the main thread (a persistent ``bpy.app.timers`` timer; tests call it
directly), applies the events to ``STATE`` and, when a job is finished, calls the handler
registered for its kind (put the picture into the gallery, import the mesh). Every failure
ends up as a message in the panel instead of an exception.

The same timer asks the server for its status now and then, so the panel notices a server
started or stopped elsewhere (the Studio shortcut, its console window) without a click.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import bpy

from .client import ServerError

POLL_S = 0.5
ACTIVE = frozenset({"submitting", "queued", "running", "downloading", "applying"})
KEEP_FINISHED = 6
KEEP_ERRORS = 5
SERVER_KEY = "__server__"
AUTO_CHECK = True
AUTO_ONLINE_S = 30.0
AUTO_OFFLINE_S = 6.0


@dataclass
class UIJob:
    key: str
    kind: str  # txt2img | img2mesh | rig | anim
    title: str
    status: str = "submitting"
    progress: float = 0.0
    stage: str = "отправка на сервер"
    server_id: str = ""
    error: str = ""
    warnings: list = field(default_factory=list)
    result: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)  # role -> local path
    meta: dict = field(default_factory=dict)  # what the handler needs (prompt, size, ...)
    output: str = ""  # what the handler made (image or object name)
    cancel_requested: bool = False
    created: float = field(default_factory=time.time)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


class _State:
    def __init__(self):
        self.jobs: OrderedDict[str, UIJob] = OrderedDict()
        self.errors: list[str] = []
        self.server = {"ok": None, "offline": False, "message": "Связь с сервером не проверялась", "health": None,
                       "time": 0.0}


STATE = _State()
_events: "queue.Queue[tuple[str, dict]]" = queue.Queue()
_handlers: dict = {}
_alive = threading.Event()
_checking = threading.Event()  # a status request is under way
_auto = {"last": 0.0}


def handler(kind: str):
    """Register the main-thread function that applies a finished job of this kind."""

    def deco(fn):
        _handlers[kind] = fn
        return fn

    return deco


def jobs_of(kind: str) -> list[UIJob]:
    return [j for j in STATE.jobs.values() if j.kind == kind]


def _post(key: str, **changes) -> None:
    _events.put((key, changes))


# -- threads ---------------------------------------------------------------------------------
def start(kind: str, title: str, client, submit, roles: tuple[str, ...], folder: Path, meta: dict | None = None) -> UIJob:
    """Run ``submit(client)`` -> poll -> download ``roles`` into ``folder`` in a thread."""
    job = UIJob(key=uuid.uuid4().hex[:8], kind=kind, title=title, meta=meta or {})
    STATE.jobs[job.key] = job
    _alive.set()
    threading.Thread(target=_run, args=(job, client, submit, roles, Path(folder)), name=f"dronegen-{kind}", daemon=True).start()
    ensure_timer()
    return job


def _run(job: UIJob, client, submit, roles, folder: Path) -> None:
    key = job.key
    try:
        server_id = submit(client)["job_id"]
        _post(key, server_id=server_id, status="queued", stage="в очереди")
        cancel_sent = False
        while True:
            if not _alive.is_set():
                return
            if job.cancel_requested and not cancel_sent:
                try:
                    client.cancel(server_id)
                except ServerError:
                    pass
                cancel_sent = True
            info = client.job(server_id)
            status = info["status"]
            if status in ("done", "error", "cancelled"):
                break
            stage = info.get("stage", "")
            if status == "queued" and info.get("queue_position"):
                stage = f"в очереди, перед вами задач: {info['queue_position']}"
            _post(key, status=status, progress=float(info.get("progress", 0.0)), stage=stage, warnings=info.get("warnings", []))
            time.sleep(POLL_S)
        if status == "error":
            _post(key, status="error", error=info.get("error") or "ошибка на сервере", warnings=info.get("warnings", []))
            return
        if status == "cancelled":
            _post(key, status="cancelled", stage="отменено")
            return
        _post(key, status="downloading", stage="скачивание результата", progress=0.99)
        files = {}
        for role in roles:
            name = info.get("files", {}).get(role)
            if name:
                files[role] = str(client.download(name, folder / name))
        if not files:
            raise ServerError("Сервер не вернул файл результата")
        _post(key, status="applying", stage="загрузка в Blender", files=files, result=info.get("result", {}),
              metrics=info.get("metrics", {}), warnings=info.get("warnings", []))
    except ServerError as e:
        _post(key, status="error", error=str(e))
    except Exception as e:  # noqa: BLE001 - a thread must never die silently
        traceback.print_exc()
        _post(key, status="error", error=f"Ошибка аддона: {type(e).__name__}: {e}")


def run(kind: str, title: str, client, work, meta: dict | None = None) -> UIJob:
    """Run ``work(client) -> dict`` in a thread (quick requests, not server jobs); the handler of ``kind`` applies
    the dict on the main thread, as for server jobs."""
    job = UIJob(key=uuid.uuid4().hex[:8], kind=kind, title=title, stage="запрос к серверу", meta=meta or {})
    STATE.jobs[job.key] = job
    _alive.set()

    def go():
        try:
            _post(job.key, status="applying", stage="применяю", progress=0.9, result=work(client))
        except ServerError as e:
            _post(job.key, status="error", error=str(e))
        except Exception as e:  # noqa: BLE001 - a thread must never die silently
            traceback.print_exc()
            _post(job.key, status="error", error=f"Ошибка аддона: {type(e).__name__}: {e}")

    threading.Thread(target=go, name=f"dronegen-{kind}", daemon=True).start()
    ensure_timer()
    return job


def _online(health: dict) -> dict:
    return {"ok": True, "offline": False, "health": health, "message": summary(health)}


def _failed(e: ServerError, prefix: str = "") -> dict:
    return {"ok": False, "offline": e.offline, "health": None, "message": f"{prefix}{e}"}


def check_server(client, quiet: bool = False) -> None:
    """Ask /health in a thread; the answer shows up in STATE.server.

    ``quiet``: the periodic check, which leaves the status line alone until the answer comes.
    """

    def work():
        try:
            _post(SERVER_KEY, **_online(client.health()))
        except ServerError as e:
            _post(SERVER_KEY, **_failed(e))
        except Exception as e:  # noqa: BLE001
            _post(SERVER_KEY, ok=False, offline=False, health=None, message=f"Ошибка проверки: {e}")
        finally:
            _checking.clear()

    _alive.set()
    _checking.set()
    if not quiet and not STATE.server["ok"]:
        STATE.server.update(ok=None, offline=False, message="Проверяю связь с сервером…")
    threading.Thread(target=work, name="dronegen-health", daemon=True).start()
    ensure_timer()


def wait_for_server(client, timeout: float = 180.0) -> None:
    """After «Запустить сервер»: poll /health until the new process answers."""

    def work():
        t0 = time.time()
        while _alive.is_set() and time.time() - t0 < timeout:
            try:
                _post(SERVER_KEY, **_online(client.health()))
                return
            except ServerError:
                _post(SERVER_KEY, ok=None, offline=False, message=f"Сервер запускается… {int(time.time() - t0)} с")
                time.sleep(2.0)
        _post(SERVER_KEY, ok=False, offline=False, health=None,
              message="Сервер не ответил за 3 минуты: посмотрите его окно или runtime/logs/server.log")

    _alive.set()
    threading.Thread(target=work, name="dronegen-wait", daemon=True).start()
    ensure_timer()


def run_and_refresh(client, action, failure: str) -> None:
    """Call ``action(client)`` in a thread, then refresh the server status."""

    def work():
        try:
            action(client)
            _post(SERVER_KEY, **_online(client.health()))
        except ServerError as e:
            _post(SERVER_KEY, **_failed(e, f"{failure}: "))

    _alive.set()
    threading.Thread(target=work, name="dronegen-action", daemon=True).start()
    ensure_timer()


def summary(health: dict) -> str:
    gpu = health.get("gpu", {})
    head = f"Сервер {health.get('version', '?')} работает"
    if gpu.get("available"):
        return f"{head} · {gpu['name']}, свободно {gpu['free_gb']:.1f} из {gpu['total_gb']:.0f} ГБ видеопамяти"
    return f"{head} · без видеокарты: {gpu.get('reason', 'нет GPU')}"


def backend_info(kind: str, name: str) -> dict | None:
    health = STATE.server.get("health") or {}
    return next((b for b in health.get("backends", {}).get(kind, []) if b["name"] == name), None)


# -- main thread ----------------------------------------------------------------------------
def tick() -> float:
    """Apply what the threads reported. Returns the delay until the next call."""
    changed = False
    while True:
        try:
            key, changes = _events.get_nowait()
        except queue.Empty:
            break
        changed = True
        if key == SERVER_KEY:
            STATE.server.update(changes, time=time.time())
            continue
        job = STATE.jobs.get(key)
        if job is None:
            continue
        for name, value in changes.items():
            setattr(job, name, value)
        if job.status == "applying":
            _apply(job)
        elif job.status == "error":
            _refresh_status()  # the server may have gone away
        if job.status == "error":
            remember_error(f"{job.title}: {job.error}")
    if changed:
        _prune()
        redraw()
    _auto_check()
    return 0.2 if any(j.active for j in STATE.jobs.values()) else 1.0


def _auto_check() -> None:
    """Refresh the status line every few seconds while the server is down, rarely while it works."""
    if not AUTO_CHECK or bpy.app.background or _checking.is_set():
        return
    server, now = STATE.server, time.time()
    interval = AUTO_ONLINE_S if server["ok"] else AUTO_OFFLINE_S
    if now - max(server["time"], _auto["last"]) < interval:  # also quiet while a start reports progress
        return
    _auto["last"] = now
    try:
        from . import prefs

        client = prefs.client()
    except Exception:  # noqa: BLE001 - a bad address is reported by the operators
        return
    if client.local or getattr(bpy.app, "online_access", True):
        check_server(client, quiet=True)


def _apply(job: UIJob) -> None:
    fn = _handlers.get(job.kind)
    try:
        job.output = (fn(job) if fn else "") or ""
        job.status, job.progress, job.stage = "done", 1.0, "готово"
    except Exception as e:  # noqa: BLE001 - shown in the panel, Blender keeps running
        traceback.print_exc()
        job.status, job.error = "error", f"Не удалось загрузить результат в Blender: {e}"
    _refresh_status()


def _refresh_status() -> None:
    """Free VRAM and loaded models change after a job: keep the status line current."""
    try:
        from . import prefs

        check_server(prefs.client())
    except Exception:  # noqa: BLE001 - a bad address is reported by the operators
        pass


def forget(key: str) -> None:
    """Remove a finished job from the panel."""
    job = STATE.jobs.get(key)
    if job is not None and not job.active:
        del STATE.jobs[key]


def remember_error(message: str) -> None:
    if message not in STATE.errors:
        STATE.errors.append(message)
        del STATE.errors[:-KEEP_ERRORS]
    print("DroneGen:", message)


def _prune() -> None:
    finished = [k for k, j in STATE.jobs.items() if not j.active]
    for k in finished[:-KEEP_FINISHED]:
        del STATE.jobs[k]


def redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "PREFERENCES", "IMAGE_EDITOR"}:
                area.tag_redraw()


def _timer():
    try:
        return tick()
    except Exception:  # noqa: BLE001 - a dead timer would stop all updates
        traceback.print_exc()
        return 1.0


def ensure_timer() -> None:
    if not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=0.1, persistent=True)


def register():
    _alive.set()
    _auto["last"] = 0.0
    ensure_timer()  # its first tick asks the server for its status


def unregister():
    _alive.clear()
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
