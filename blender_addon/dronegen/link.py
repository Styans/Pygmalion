"""«В Blender» from the Studio: models in the server's inbox are imported into the DroneGen collection.

A timer on the main thread starts a short background thread every few seconds. The thread asks
the server for new items, claims one, downloads its GLB and queues it; the import itself (``bpy``)
runs on the main thread in the next ticks, and the result is reported back to the server in
another short thread, so the Studio chat can say «в сцене: DG Домбра».

The thread never touches ``bpy``; a server that is down only slows the polling down.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path

import bpy

from . import importer, jobs, prefs
from .client import Client, ServerError

POLL_S = 2.0
SLOW_S = 10.0  # while the server does not answer
PREFIX = "DG "

_ready: "queue.Queue[tuple]" = queue.Queue()
_state = {"busy": False, "next": 0.0, "ok": None, "imported": []}


def settings_snapshot() -> tuple | None:
    """(url, token, timeout, cache folder, size) from the add-on settings, read on the main thread."""
    try:
        p = prefs.get()
        if not p.studio_link:
            return None
        scene = bpy.context.scene
        size = scene.dronegen.target_size if scene is not None and hasattr(scene, "dronegen") else 2.0
        return p.server_url, p.token, float(p.timeout), prefs.cache_dir() / "studio", size
    except Exception:  # noqa: BLE001 - add-on not fully registered yet
        return None


def _poll(url: str, token: str, timeout: float, folder: Path, size: float) -> None:
    """One round in a thread: claim and download what is waiting."""
    try:
        client = Client(url, token, timeout)
        items = client.inbox()
        _state["ok"] = True
        for item in items[:3]:
            try:
                client.inbox_taken(item["id"])
                path = client.download(item["file"], folder / f"{item['id']}_{item['file']}")
            except ServerError as e:
                _report(client, item["id"], error=f"не удалось забрать файл: {e}")
                continue
            _ready.put((item, path, size, client))
    except ServerError:
        _state["ok"] = False
    except Exception:  # noqa: BLE001 - a thread must not die with a traceback in the user's console
        traceback.print_exc()
        _state["ok"] = False
    finally:
        _state["busy"] = False
        _state["next"] = time.time() + (POLL_S if _state["ok"] else SLOW_S)


def _report(client: Client, item_id: str, obj: str = "", error: str = "") -> None:
    def work():
        try:
            client.inbox_done(item_id, obj, error)
        except Exception:  # noqa: BLE001 - the model is in the scene anyway
            traceback.print_exc()

    threading.Thread(target=work, name="dronegen-inbox-done", daemon=True).start()


def frame_selected(context) -> None:
    """Bring the new object into view in the first 3D viewport (no-op without a window)."""
    wm = bpy.context.window_manager
    for window in (wm.windows if wm is not None else []):
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            try:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    bpy.ops.view3d.view_selected()
            except Exception:  # noqa: BLE001 - cosmetic
                pass
            return


def apply_ready(context=None) -> list[str]:
    """Main thread: import what the thread downloaded. Returns the names of the new objects."""
    context = context or bpy.context
    names = []
    while True:
        try:
            item, path, size, client = _ready.get_nowait()
        except queue.Empty:
            break
        try:
            info = {"source": "studio", "inbox": item["id"], "file": item["file"]}
            obj = importer.import_glb(context, Path(path), PREFIX + item["name"], size, info)
            names.append(obj.name)
            _state["imported"].append(obj.name)
            del _state["imported"][:-5]
            frame_selected(context)
            _report(client, item["id"], obj=obj.name)
        except Exception as e:  # noqa: BLE001 - reported to the Studio chat and the panel
            traceback.print_exc()
            _report(client, item["id"], error=str(e)[:200])
            jobs.remember_error(f"Студия → Blender: не удалось импортировать «{item['name']}»: {e}")
        jobs.redraw()
    return names


def tick() -> float:
    """Timer: import what is ready, start the next poll when it is due."""
    apply_ready()
    if not _state["busy"] and time.time() >= _state["next"]:
        snap = settings_snapshot()
        if snap is None:
            _state["next"] = time.time() + SLOW_S
        else:
            _state["busy"] = True
            threading.Thread(target=_poll, args=snap, name="dronegen-inbox", daemon=True).start()
    return 0.5


def _timer():
    try:
        return tick()
    except Exception:  # noqa: BLE001 - a dead timer would stop the link
        traceback.print_exc()
        return 2.0


def register():
    _state.update(busy=False, next=time.time() + 3.0, ok=None)
    if not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=1.0, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
