"""Open the DroneGen Studio (the chat app) in its own window: Alt+Shift+D or a button.

The Studio is a web app served by the DroneGen server. It opens as an application window
(Edge or Chrome with ``--app``: no tabs, no address bar), with its own browser profile so
that it never touches the user's browsing; any other browser is the fallback.
If the server is not running it is started first (the same as the «Запустить сервер» button).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import bpy

from . import jobs, prefs, server_proc
from .client import Client, ServerError

WAIT_S = 180.0
KEYMAP = ("D", "PRESS", True, True)  # Alt+Shift+D: free in Blender's default keymap
_keymaps: list = []


def browser_candidates() -> list[str]:
    """Chromium-based browsers that can show a page as an application window, best first."""
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
        found = [Path(r) / sub for r in roots if r
                 for sub in ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")]
        return [str(p) for p in found if p.is_file()]
    if sys.platform == "darwin":
        apps = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
        return [a for a in apps if Path(a).is_file()]
    names = ("microsoft-edge", "google-chrome", "chromium", "chromium-browser")
    return [p for p in (shutil.which(n) for n in names) if p]


def open_window(url: str, profile: Path) -> str:
    """Show ``url`` as an app window; returns how it was opened (for the report)."""
    for exe in browser_candidates():
        profile.mkdir(parents=True, exist_ok=True)
        cmd = [exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1500,950",
               "--no-first-run", "--no-default-browser-check"]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            return Path(exe).stem
        except OSError:
            continue
    webbrowser.open(url)
    return "браузер по умолчанию"


def open_when_ready(url: str, token: str, timeout: float, home: str, profile: Path, wait: float = WAIT_S) -> None:
    """Thread body: make sure the server answers (start it if it is local), then open the window.

    Problems are posted to the panel through ``jobs`` (no ``bpy`` here).
    """
    post = lambda **kw: jobs._post(jobs.SERVER_KEY, **kw)  # noqa: E731
    try:
        client = Client(url, token, timeout)
        studio_url = client.studio_url()
    except ServerError as e:
        post(ok=False, message=str(e))
        return
    try:
        client.health()
    except ServerError as first:
        if not client.local:
            post(ok=False, message=str(first))
            return
        try:
            post(ok=None, message=server_proc.start(home, url))
        except RuntimeError as e:
            post(ok=False, message=str(e))
            return
        t0 = time.time()
        while True:
            try:
                client.health()
                break
            except ServerError:
                if time.time() - t0 > wait:
                    post(ok=False, message="Сервер не ответил за 3 минуты: посмотрите его окно или runtime/logs/server.log")
                    return
                post(ok=None, message=f"Сервер запускается… {int(time.time() - t0)} с")
                time.sleep(1.5)
    how = open_window(studio_url, profile)
    post(ok=True, message=f"Студия открыта ({how})")


def open_async() -> str | None:
    """Main thread: start opening the Studio. Returns an error text, or None when it is under way."""
    try:
        p = prefs.get()
        profile = prefs.cache_dir() / "studio-browser"
        args = (p.server_url, p.token, float(p.timeout), p.server_home, profile)
        Client(p.server_url, p.token)  # a bad address is reported now
    except ServerError as e:
        return str(e)
    jobs._alive.set()
    threading.Thread(target=open_when_ready, args=args, name="dronegen-studio", daemon=True).start()
    jobs.ensure_timer()
    return None


# -- the hotkey ----------------------------------------------------------------------------
def _keymap_item():
    """The hotkey as the user has it now (their keymap edits live in the «user» key configuration)."""
    wm = bpy.context.window_manager
    for kc in (wm.keyconfigs.user, wm.keyconfigs.addon) if wm is not None else ():
        km = kc.keymaps.get("Window") if kc is not None else None
        kmi = next((k for k in km.keymap_items if k.idname == "dronegen.open_studio"), None) if km else None
        if kmi is not None:
            return kc, km, kmi
    return None


def hotkey_text() -> str:
    found = _keymap_item()
    if found is None or not found[2].active:
        return ""
    return found[2].to_string()


def draw_hotkey(layout) -> None:
    """The hotkey editor of the add-on preferences (the same widget as in Preferences → Keymap)."""
    found = _keymap_item()
    if found is None:
        layout.label(text="Горячая клавиша не назначена: Preferences → Keymap → Window → «Открыть студию»", icon="INFO")
        return
    kc, km, kmi = found
    try:
        import rna_keymap_ui

        layout.context_pointer_set("keymap", km)
        rna_keymap_ui.draw_kmi([], kc, km, kmi, layout, 0)
    except Exception:  # noqa: BLE001 - Blender's own helper module changed: a plain line is enough
        layout.label(text=f"Горячая клавиша: {kmi.to_string()} (Preferences → Keymap → Window)", icon="EVENT_D")


def register():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon if wm is not None else None
    if kc is None:  # background mode: no keymaps
        return
    km = kc.keymaps.new(name="Window", space_type="EMPTY", region_type="WINDOW")
    key, value, alt, shift = KEYMAP
    kmi = km.keymap_items.new("dronegen.open_studio", key, value, alt=alt, shift=shift)
    _keymaps.append((km, kmi))


def unregister():
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except (RuntimeError, ReferenceError):
            pass
    _keymaps.clear()
