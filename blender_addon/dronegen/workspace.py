"""The «DroneGen» workspace: pictures on the left, the 3D view on the right, outliner, properties.

It is appended from a small .blend that a background Blender builds on first use
(``workspace_template.py``) and that is cached per Blender version. Switching to the workspace
never changes the scene: its mode is Object mode.
"""

from __future__ import annotations

import subprocess
import threading
import traceback
from pathlib import Path

import bpy

from . import jobs, prefs

NAME = "DroneGen"
BUILD_TIMEOUT = 120
_building = threading.Event()
_ready: list[Path] = []  # built template paths waiting for the main thread


def template_path() -> Path:
    return prefs.cache_dir() / f"workspace-{bpy.app.version_string.replace(' ', '_')}.blend"


def build_template(path: Path) -> None:
    """Build the template in a background Blender (blocking; run it in a thread)."""
    script = Path(__file__).with_name("workspace_template.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.blend")
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python", str(script), "--", str(tmp)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=BUILD_TIMEOUT)
    if done.returncode != 0 or not tmp.is_file():
        tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
        raise RuntimeError("не удалось собрать воркспейс: " + " | ".join(tail))
    tmp.replace(path)


def _build_async(path: Path) -> None:
    def work():
        try:
            build_template(path)
            _ready.append(path)
        except Exception as e:  # noqa: BLE001 - shown in the panel
            traceback.print_exc()
            jobs._post(jobs.SERVER_KEY, ok=False, message=f"Воркспейс DroneGen: {e}")
        finally:
            _building.clear()

    _building.set()
    threading.Thread(target=work, name="dronegen-workspace", daemon=True).start()


def activate(context, path: Path | None = None) -> bool:
    """Switch to the workspace, appending it first if it is not in this file. Main thread."""
    window = context.window or (bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None)
    if window is None:
        return False
    ws = bpy.data.workspaces.get(NAME)
    if ws is None:
        path = path or template_path()
        if not path.is_file():
            return False
        with bpy.context.temp_override(window=window):
            bpy.ops.workspace.append_activate(idname=NAME, filepath=str(path))
        ws = bpy.data.workspaces.get(NAME)
        if ws is None:
            return False
    window.workspace = ws
    return True


def open_workspace(context) -> str | None:
    """The «Открыть воркспейс» button. Returns a message when the workspace is being built."""
    if activate(context):
        return None
    if _building.is_set():
        return "Воркспейс DroneGen собирается, подождите несколько секунд…"
    _build_async(template_path())
    jobs.ensure_timer()
    bpy.app.timers.register(_finish, first_interval=0.5)
    return "Собираю воркспейс DroneGen (первый раз — несколько секунд)…"


def _finish():
    """Timer: switch as soon as the template is built."""
    if _ready:
        path = _ready.pop()
        try:
            activate(bpy.context, path)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            jobs.remember_error("Воркспейс DroneGen: не удалось подключить собранный файл")
        jobs.redraw()
        return None
    return 0.5 if _building.is_set() else None
