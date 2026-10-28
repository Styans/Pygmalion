"""Open the Studio as an application window (``python -m dronegen_server --open-studio``, start_studio.bat).

Edge or Chrome with ``--app`` shows the page without tabs and an address bar; the window has its
own browser profile in ``<home>/studio/browser`` and never touches the user's browsing. Any other
browser is the fallback. (Blender's add-on has its own copy of this: it cannot import the server.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


def browser_candidates() -> list[str]:
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
        found = [Path(r) / sub for r in roots if r
                 for sub in ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")]
        return [str(p) for p in found if p.is_file()]
    if sys.platform == "darwin":
        apps = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
        return [a for a in apps if Path(a).is_file()]
    return [p for p in (shutil.which(n) for n in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser")) if p]


def spawn_window(url: str, profile: Path) -> tuple[str, subprocess.Popen | None]:
    """Show ``url`` as an app window; returns the browser name and its process (None for the default browser)."""
    for exe in browser_candidates():
        profile.mkdir(parents=True, exist_ok=True)
        cmd = [exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1500,950",
               "--no-first-run", "--no-default-browser-check"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            return Path(exe).stem, proc
        except OSError:
            continue
    webbrowser.open(url)
    return "браузер по умолчанию", None


def open_window(url: str, profile: Path) -> str:
    return spawn_window(url, profile)[0]


def _window_processes(profile: Path) -> list:
    """Browser processes that use the Studio's own profile folder (the window is theirs)."""
    import psutil

    needle = str(profile).lower()
    found = []
    for proc in psutil.process_iter(["cmdline"]):
        try:
            line = " ".join(proc.info["cmdline"] or []).lower()
        except psutil.Error:
            continue
        if needle in line and "--user-data-dir" in line:
            found.append(proc)
    return found


def wait_window_closed(profile: Path, appear_s: float = 20.0, poll_s: float = 1.5) -> bool:
    """Block until the Studio window is closed; False if no window was ever seen (e.g. the default browser).

    Edge and Chrome hand the window to a different process, so the one we started cannot be waited for:
    we watch for any process of the window's private profile instead.
    """
    import time

    t0 = time.time()
    while not _window_processes(profile):
        if time.time() - t0 > appear_s:
            return False
        time.sleep(0.3)
    while _window_processes(profile):
        time.sleep(poll_s)
    return True
