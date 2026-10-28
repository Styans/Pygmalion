"""End-to-end check in headless Blender: server -> add-on -> picture -> mesh in the scene.

    runtime\\venv\\Scripts\\python tests\\run_blender_e2e.py            # mock + silhouette (no GPU)
    runtime\\venv\\Scripts\\python tests\\run_blender_e2e.py --real     # + Z-Image-Turbo and TripoSR
    runtime\\venv\\Scripts\\python tests\\run_blender_e2e.py --repo     # the server code of this folder, not the installed one

Blender gets its own profile in runtime/test/ (BLENDER_USER_RESOURCES), so the user's Blender
settings and add-ons are never touched. The add-on is built and installed into that profile
with Blender's extension tools, the same way install.py does it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
ADDON = ROOT / "blender_addon" / "dronegen"


def venv_python(home: Path) -> Path:
    return home / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_blender(arg: str | None) -> Path:
    candidates = [arg, os.environ.get("DRONEGEN_BLENDER"), shutil.which("blender")]
    if sys.platform == "win32":
        candidates += [str(p) for p in sorted(Path(r"C:\Program Files\Blender Foundation").glob("Blender*/blender.exe"), reverse=True)]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise SystemExit("Blender не найден: укажите --blender путь\\к\\blender.exe")


def run(cmd, env, timeout=600):
    r = subprocess.run([str(c) for c in cmd], env=env, timeout=timeout, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"команда завершилась с ошибкой {r.returncode}: {cmd[:6]}")
    return r


def wait_health(port: int, proc, timeout=90.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("сервер завершился при запуске, см. лог")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.load(r).get("name") == "dronegen":
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("сервер не ответил за 90 с")


def run_gui(blender: Path, env: dict, port: int, test_dir: Path) -> bool:
    """A Blender with a window (in the same isolated profile): what a background run cannot see."""
    out = test_dir / "gui"
    cmd = [blender, "--no-window-focus", "--window-geometry", "40", "40", "1560", "940", "--python-exit-code", "1",
           "--python", ROOT / "tests" / "blender_gui_smoke.py", "--", "--url", f"http://127.0.0.1:{port}",
           "--cache", test_dir / "cache_gui", "--out", out]
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=400)
    (test_dir / "blender_gui.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    report = out / "report.json"
    results = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else []
    for res in results:
        print(f"  [{'OK ' if res['ok'] else 'ОШИБКА'}] окно Blender: {res['name']} {res['info'][:140]}")
    keep = RUNTIME / "test" / "gui_smoke"
    shutil.rmtree(keep, ignore_errors=True)
    if out.is_dir():
        shutil.copytree(out, keep)
    good = r.returncode == 0 and bool(results) and all(x["ok"] for x in results)
    print(f"Blender с окном: код {r.returncode}, {time.time() - t0:.0f} с; проверок {len(results)}; скриншоты: {keep}")
    if not good:
        print(r.stdout[-3000:], r.stderr[-2000:])
    return good


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--blender", default=None)
    p.add_argument("--real", action="store_true", help="также Z-Image-Turbo и TripoSR (нужны GPU и веса)")
    p.add_argument("--keep", action="store_true", help="не удалять папку теста")
    p.add_argument("--gui", action="store_true", help="ещё и Blender с окном: воркспейс, модель из студии, скриншоты (runtime/test/gui_smoke)")
    p.add_argument("--home", type=Path, default=RUNTIME, help="установка, чей сервер проверять (по умолчанию runtime)")
    p.add_argument("--repo", action="store_true", help="код сервера из папки проекта, а не установленный (Python и веса — из --home)")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Russian and arrows survive pipes on Windows
        stream.reconfigure(encoding="utf-8", errors="replace")

    blender = find_blender(args.blender)
    test_dir = RUNTIME / "test" / time.strftime("e2e-%Y%m%d-%H%M%S")
    test_dir.mkdir(parents=True)
    # made beforehand: into a profile folder that is not there yet, Blender copies the user's own settings
    (test_dir / "blender").mkdir()
    env = {**os.environ, "BLENDER_USER_RESOURCES": str(test_dir / "blender"), "PYTHONUTF8": "1"}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    print(f"Blender: {blender}\nПапка теста: {test_dir}")

    stage = test_dir / "stage" / "dronegen"
    shutil.copytree(ADDON, stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "install_info.json"))
    run([blender, "--factory-startup", "-b", "--command", "extension", "build", "--source-dir", stage, "--output-dir", test_dir], env)
    package = next(test_dir.glob("dronegen-*.zip"))
    run([blender, "-b", "--command", "extension", "install-file", "-r", "user_default", "-e", package], env)
    print(f"Аддон собран и установлен в тестовый профиль: {package.name}")

    port = free_port()
    backends = "mock,silhouette,template" + (",zimage_turbo,triposr" if args.real else "")
    server_log = open(test_dir / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        [str(venv_python(args.home)), "-m", "dronegen_server", "--home", str(test_dir / "server"), "--models-dir", str(args.home / "models"),
         "--port", str(port), "--backends", backends, "--idle-unload", "0"],
        stdout=server_log, stderr=subprocess.STDOUT, env={**env, "PYTHONPATH": str(ROOT / "server")} if args.repo else env,
    )
    ok = False
    try:
        wait_health(port, server)
        print(f"Сервер: http://127.0.0.1:{port} ({backends})")
        report = test_dir / "report.json"
        cmd = [blender, "-b", "--python-exit-code", "1", "--python", ROOT / "tests" / "blender_e2e.py", "--",
               "--url", f"http://127.0.0.1:{port}", "--cache", test_dir / "cache", "--report", report,
               "--dead-port", str(free_port())] + (["--real"] if args.real else [])
        t0 = time.time()
        r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
        (test_dir / "blender.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        results = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else []
        for res in results:
            mark = "OK " if res["ok"] else "ОШИБКА"
            extra = f" — {res['error']}" if not res["ok"] else (f" ({res['info']})" if res.get("info") else "")
            print(f"  [{mark}] {res['name']}: {res['seconds']:.1f} с{extra}")
        ok = r.returncode == 0 and results and all(x["ok"] for x in results)
        print(f"Blender: код {r.returncode}, {time.time() - t0:.0f} с; проверок {len(results)}")
        if not ok:
            print(r.stdout[-6000:], r.stderr[-3000:])
        if ok and args.gui:
            ok = run_gui(blender, env, port, test_dir)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
    if ok and not args.keep:
        shutil.rmtree(test_dir, ignore_errors=True)
    print("ИТОГ:", "все проверки пройдены" if ok else f"есть ошибки, см. {test_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
