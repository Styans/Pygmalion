"""Start and stop the local server from Blender. It runs as its own process (own console)."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

from .client import is_local

_proc: subprocess.Popen | None = None


def venv_python(home: Path) -> Path:
    return home / ("venv/Scripts/python.exe" if sys.platform == "win32" else "venv/bin/python")


def start(home: str, url: str) -> str:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return "Сервер уже запущен из Blender — дождитесь, пока он загрузится"
    from . import prefs

    installed_home = prefs. _install_info().get("home", "")
    home_path = Path(home).expanduser().resolve() if home else None
    if (home_path is None or not venv_python(home_path).is_file()) and installed_home:
        candidate = Path(installed_home).expanduser().resolve()
        if venv_python(candidate).is_file():
            home_path = candidate
            try:
                prefs.get().server_home = str(candidate)
            except Exception:  # noqa: BLE001
                pass
    if home_path is None:
        raise RuntimeError("Не указана папка сервера: Edit → Preferences → Add-ons → DroneGen → Папка сервера")
    py = venv_python(home_path)
    if not py.is_file():
        raise RuntimeError(f"Не найден {py}. Установите сервер: python install.py (в папке проекта)")
    parts = urllib.parse.urlsplit(url)
    if not is_local(parts.hostname):
        raise RuntimeError("Из Blender можно запустить только локальный сервер (адрес 127.0.0.1)")
    port = parts.port or 8188
    args = [str(py), "-m", "dronegen_server", "--home", str(home_path), "--port", str(port)]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}  # not Blender's Python
    env["PYTHONUTF8"] = "1"
    if sys.platform == "win32":
        _proc = subprocess.Popen(args, cwd=home_path, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        log_dir = home_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / "server_console.log", "ab")
        _proc = subprocess.Popen(args, cwd=home_path, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return f"Сервер запускается ({url}). Первый запуск с моделями занимает до минуты"


def started_here() -> bool:
    return _proc is not None and _proc.poll() is None


def stop(client) -> str:
    global _proc
    try:
        client.shutdown()
        msg = "Сервер останавливается"
    except Exception as e:  # noqa: BLE001 - fall back to killing the process we started
        if not started_here():
            raise RuntimeError(f"Не удалось остановить сервер: {e}") from e
        msg = "Сервер остановлен"
    if started_here():
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.terminate()
    _proc = None
    return msg
