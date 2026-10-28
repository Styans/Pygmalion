"""Start the server:  python -m dronegen_server [--port 8188] [--home DIR] ...

Settings come from ``<home>/config.json`` (written by install.py), flags override them.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import socket
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .config import Settings, utf8_stdio


def _setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    settings.logs.mkdir(parents=True, exist_ok=True)
    file = logging.handlers.RotatingFileHandler(
        settings.logs / "server.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(fmt)
    root.addHandler(file)


def _port_problem(host: str, port: int) -> str | None:
    """None if the port is free, otherwise a message for the user."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return None
        except OSError:
            pass
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            if json.load(r).get("name") == "dronegen":
                return f"Сервер DroneGen уже запущен на порту {port} — второй не нужен."
    except Exception:  # noqa: BLE001 - something else is listening
        pass
    return (
        f"Порт {port} занят другой программой (8188 использует и ComfyUI). "
        f"Запустите с другим портом, например: start_server --port 8189, "
        f"и укажите адрес http://127.0.0.1:8189 в настройках аддона."
    )


def _studio_url(settings) -> str:
    import urllib.parse

    host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    url = f"http://{host}:{settings.port}/studio/"
    return url + "?token=" + urllib.parse.quote(settings.token) if settings.token else url


def _open_studio_when_ready(server, settings, log, quit_with_window: bool = False) -> None:
    """Open the window once the server is up; with ``quit_with_window`` closing it stops the server (app mode)."""
    import threading
    import time

    from .launcher import spawn_window, wait_window_closed

    def work():
        for _ in range(600):  # up to 60 s
            if server.started:
                break
            time.sleep(0.1)
        else:
            return
        url = _studio_url(settings)
        profile = settings.home / "studio" / "browser"
        name, proc = spawn_window(url, profile)
        log.info("Студия открыта (%s): %s", name, url)
        # app mode: closing the window stops the server. Without a window we can watch (the default browser)
        # the server keeps running; «Выключить сервер» is in the Studio settings
        if quit_with_window and proc is not None and wait_window_closed(profile):
            log.info("Окно студии закрыто — останавливаю сервер")
            server.should_exit = True

    threading.Thread(target=work, name="open-studio", daemon=True).start()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dronegen_server", description=f"DroneGen server {__version__}")
    p.add_argument("--home", type=Path, default=None, help="папка установки (venv, models, outputs)")
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--host", default=None, help="127.0.0.1 — только этот компьютер; 0.0.0.0 — вся сеть")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--backends", default=None, help="через запятую, например mock,silhouette (по умолчанию все)")
    p.add_argument("--online", action="store_true", help="разрешить загрузку из интернета (по умолчанию офлайн)")
    p.add_argument("--token", default=None, help="токен доступа (для запуска не на localhost)")
    p.add_argument("--idle-unload", type=float, default=None, help="выгружать модели после N секунд простоя")
    p.add_argument("--log-level", default=None)
    p.add_argument("--open-studio", action="store_true", help="открыть окно студии, когда сервер готов")
    p.add_argument("--app", action="store_true",
                   help="режим приложения (ярлык): без консоли, окно студии; закрыли окно — сервер остановился. "
                        "Если сервер уже работает, просто открывает окно")
    p.add_argument("--make-shortcuts", action="store_true", help="создать ярлык «DroneGen Studio» (рабочий стол и меню Пуск)")
    args = p.parse_args(argv)
    utf8_stdio()

    settings = Settings.load(
        args.home,
        models_dir=args.models_dir,
        host=args.host,
        port=args.port,
        device=args.device,
        backends=[b.strip() for b in args.backends.split(",") if b.strip()] if args.backends else None,
        offline=False if args.online else None,
        token=args.token,
        idle_unload_s=args.idle_unload,
        log_level=args.log_level,
    )
    settings.ensure_dirs()
    if args.make_shortcuts:
        from .shortcuts import create_shortcuts

        for line in create_shortcuts(settings.home):
            print(line)
        return 0
    if sys.stdout is None or sys.stderr is None:  # pythonw.exe has no console: keep print() and uvicorn working
        console = open(settings.logs / "app-console.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or console
        sys.stderr = sys.stderr or console
    settings.apply_env()  # before huggingface_hub / diffusers are imported
    _setup_logging(settings)
    log = logging.getLogger("dronegen")

    problem = _port_problem(settings.host, settings.port)
    if problem and args.app and problem.startswith("Сервер DroneGen уже запущен"):
        from .launcher import open_window

        open_window(_studio_url(settings), settings.home / "studio" / "browser")  # a second click just shows the window
        return 0
    if problem:
        log.error(problem)
        return 2
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.token:
        log.warning("Сервер открыт в сеть без токена: любой в сети может запускать задачи (используйте --token)")

    import uvicorn

    from .app import create_app

    app = create_app(settings)
    # open event streams (the Studio window) must not hold the server back when it is asked to stop
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level=settings.log_level, access_log=False,
                            timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    app.state.server = server
    log.info("DroneGen %s: http://%s:%d (данные: %s)", __version__, settings.host, settings.port, settings.home)
    if args.open_studio or args.app:
        _open_studio_when_ready(server, settings, log, quit_with_window=args.app)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
