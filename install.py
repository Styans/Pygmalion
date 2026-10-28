#!/usr/bin/env python3
"""DroneGen installer: the local server, the model weights and the Blender add-on in one command.

    python install.py                  # everything; GPU, CUDA and Blender are detected
    python install.py --skip-sdxl      # without the SDXL fallback (~7 GB less)
    python install.py --skip-llm       # without the local chat model (~6 GB less)
    python install.py --profile light  # no PyTorch: only the silhouette and test backends
    python install.py --dev            # run the server code from this checkout (editable)

Everything goes into ``runtime/`` next to this file (venv, Python, models, outputs, logs);
only the Blender add-on is installed into Blender itself.
Re-running is safe: finished steps are skipped and downloads resume where they stopped.
Only the Python standard library is used, any Python 3.9+ can run this file.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER_SRC = ROOT / "server"
ADDON_SRC = ROOT / "blender_addon" / "dronegen"

PYTHON_VERSION = "3.12"  # every dependency (PyMCubes, onnxruntime, torch) has wheels for it
TORCH_VERSION = "2.14.1"
TORCHVISION_VERSION = "0.29.1"  # Hunyuan3D-2 imports it; must come from the same CUDA index as torch
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
# (wheel tag, CUDA version it needs from the driver, minimum compute capability)
CUDA_BUILDS = [("cu132", (13, 2), (7, 5)), ("cu130", (13, 0), (7, 5)), ("cu126", (12, 6), (5, 0))]
# model code that is not on PyPI: pinned commits, put on the venv's sys.path with a .pth file
# (repo, commit, paths to check out; None: everything)
VENDOR = {
    "TripoSR": ("VAST-AI-Research/TripoSR", "107cefdc244c39106fa830359024f6a2f1c78871", None),
    # the repo is ~150 MB of demo assets around 0.5 MB of code
    "Hunyuan3D-2": ("Tencent/Hunyuan3D-2", "f8db63096c8282cb27354314d896feba5ba6ff8a", ("/hy3dgen/", "/LICENSE", "/NOTICE")),
}
UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download/"
DEFAULT_PORT = 8188

IS_WIN = sys.platform == "win32"
_log_file = None


# ---------------------------------------------------------------------------- output
def say(msg: str = "") -> None:
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def step(n: int, total: int, title: str) -> None:
    say(f"\n[{n}/{total}] {title}")


class InstallError(Exception):
    pass


def run(cmd: list, env: dict | None = None, check: bool = True, capture: bool = False, cwd=None) -> str:
    """Run a command, echoing its output to the console and the log."""
    shown = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    if _log_file is not None:
        _log_file.write(f"$ {shown}\n")
    merged = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        [str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=merged, cwd=cwd,
        text=True, encoding="utf-8", errors="replace",
    )
    out = []
    for line in proc.stdout:
        out.append(line)
        if not capture:
            sys.stdout.write(line)
            sys.stdout.flush()
        if _log_file is not None:
            _log_file.write(line)
    proc.wait()
    if check and proc.returncode != 0:
        tail = "".join(out[-15:])
        raise InstallError(f"Команда завершилась с ошибкой ({proc.returncode}): {shown}\n{tail}")
    return "".join(out)


def default_home() -> Path:
    """``runtime/`` next to this file: the whole install stays inside the project folder."""
    env = os.environ.get("DRONEGEN_HOME")
    return Path(env) if env else ROOT / "runtime"


# ---------------------------------------------------------------------------- detection
def detect_gpu() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        q = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
        name, total, used, driver, cap = [x.strip() for x in q.split(",")]
        head = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001 - broken driver: treat as no GPU
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", head)
    cuda = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    try:
        major, minor = (int(x) for x in cap.split("."))
    except ValueError:
        major, minor = 0, 0
    return {
        "name": name,
        "vram_gb": round(float(total) / 1024, 1),
        "vram_used_gb": round(float(used) / 1024, 1),
        "driver": driver,
        "cuda": cuda,
        "capability": (major, minor),
    }


def detect_memory() -> dict:
    gb = 1024**3
    if IS_WIN:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong)
                for n in ("total", "avail", "commit_total", "commit_avail", "vtotal", "vavail", "vext")
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return {
            "total_gb": st.total / gb, "available_gb": st.avail / gb,
            "commit_limit_gb": st.commit_total / gb, "commit_available_gb": st.commit_avail / gb,
            "pagefile_auto": pagefile_auto(),
        }
    if sys.platform == "darwin":
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0)
        return {"total_gb": total / gb, "available_gb": total / gb}
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k] = int(v.split()[0]) * 1024
    return {"total_gb": info.get("MemTotal", 0) / gb, "available_gb": info.get("MemAvailable", 0) / gb}


def pagefile_auto() -> bool:
    """Windows: a system-managed page file grows when memory runs short."""
    try:
        import winreg

        key = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            files, _ = winreg.QueryValueEx(k, "PagingFiles")
    except OSError:
        return False
    entries = files if isinstance(files, list) else [files]
    return any(e.startswith("?:") or e.strip().endswith(" 0 0") for e in entries if e)


def blender_version(exe: Path) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run([str(exe), "--factory-startup", "-b", "--version"], capture_output=True, text=True,
                             timeout=120, encoding="utf-8", errors="replace").stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(x) for x in m.groups()) if m else None


def find_blenders(extra: list[str]) -> list[tuple[Path, tuple]]:
    candidates = [Path(p) for p in extra]
    if shutil.which("blender"):
        candidates.append(Path(shutil.which("blender")))
    if IS_WIN:
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
            if base:
                candidates += sorted(Path(base, "Blender Foundation").glob("Blender*/blender.exe"))
                candidates.append(Path(base, "Steam", "steamapps", "common", "Blender", "blender.exe"))
    elif sys.platform == "darwin":
        candidates += [Path("/Applications/Blender.app/Contents/MacOS/Blender")]
    else:
        candidates += sorted(Path("/opt").glob("blender*/blender")) + [Path("/snap/bin/blender")]
    found, seen = [], set()
    for c in candidates:
        if c.is_file() and c.resolve() not in seen:
            seen.add(c.resolve())
            v = blender_version(c)
            if v:
                found.append((c, v))
    return found


def pick_cuda_build(gpu: dict | None, forced: str) -> str | None:
    if forced == "cpu":
        return None
    if forced != "auto":
        return forced
    if gpu is None:
        return None
    for tag, need, cap in CUDA_BUILDS:
        if gpu["cuda"] >= need and gpu["capability"] >= cap:
            return tag
    return None


# ---------------------------------------------------------------------------- uv and the venv
def ensure_uv(home: Path) -> Path:
    found = shutil.which("uv")
    if found:
        return Path(found)
    tools = home / "tools"
    exe = tools / ("uv.exe" if IS_WIN else "uv")
    if exe.is_file():
        return exe
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if IS_WIN:
        asset = f"uv-{arch}-pc-windows-msvc.zip"
    elif sys.platform == "darwin":
        asset = f"uv-{arch}-apple-darwin.tar.gz"
    else:
        asset = f"uv-{arch}-unknown-linux-gnu.tar.gz"
    say(f"  скачиваю uv ({asset}) с GitHub astral-sh/uv")
    tools.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(UV_RELEASE + asset, timeout=120).read()
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if Path(member).name in ("uv.exe", "uvx.exe"):
                    (tools / Path(member).name).write_bytes(zf.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if Path(member.name).name in ("uv", "uvx") and member.isfile():
                    (tools / Path(member.name).name).write_bytes(tf.extractfile(member).read())
                    (tools / Path(member.name).name).chmod(0o755)
    if not exe.is_file():
        raise InstallError("Не удалось получить uv")
    return exe


def uv_env(home: Path) -> dict:
    return {
        "UV_PYTHON_INSTALL_DIR": str(home / "python"),  # the interpreter lives with the install
        "UV_CACHE_DIR": str(home / "cache" / "uv"),
        "UV_LINK_MODE": "copy",
        "PYTHONUTF8": "1",
    }


def venv_python(home: Path) -> Path:
    return home / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def managed_python(home: Path) -> Path | None:
    """The interpreter uv unpacked into <home>/python (the exact patch-version folder)."""
    pattern = f"cpython-{PYTHON_VERSION}.*"
    for d in sorted((home / "python").glob(pattern), reverse=True):
        exe = d / ("python.exe" if IS_WIN else "bin/python3")
        if exe.is_file() and not d.is_symlink():
            return exe
    return None


def ensure_python(uv: Path, home: Path) -> Path:
    exe = managed_python(home)
    if exe is None:
        # --no-bin: no python3.x shims in ~/.local/bin. uv 0.12 on Windows may fail afterwards on its
        # "minor version link" junction (astral-sh/uv#19622) although the interpreter is unpacked:
        # that link is not needed here, the venv points at the exact folder.
        run([uv, "python", "install", PYTHON_VERSION, "--no-bin"], env=uv_env(home), check=False)
        exe = managed_python(home)
    if exe is None:
        raise InstallError(f"uv не смог установить Python {PYTHON_VERSION} в {home / 'python'}")
    return exe


def ensure_venv(uv: Path, home: Path) -> Path:
    py = venv_python(home)
    if py.is_file():
        v = subprocess.run([str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True).stdout.strip()
        if v == PYTHON_VERSION:
            say(f"  venv уже есть: {py.parent.parent} (Python {v})")
            return py
        say(f"  venv на Python {v or '?'}, нужен {PYTHON_VERSION} — пересоздаю")
        shutil.rmtree(home / "venv")
    base = ensure_python(uv, home)
    run([base, "-m", "venv", home / "venv"])
    return py


def torch_state(py: Path, module: str = "torch") -> str:
    code = f"import {module}; print({module}.__version__)"
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def install_torch(uv: Path, py: Path, home: Path, tag: str | None) -> None:
    want = f"{TORCH_VERSION}+{tag}" if tag else TORCH_VERSION
    want_vision = f"{TORCHVISION_VERSION}+{tag}" if tag else TORCHVISION_VERSION
    have, have_vision = torch_state(py), torch_state(py, "torchvision")
    ok = lambda have, want: have == want or (tag is None and have.startswith(want))  # noqa: E731
    if ok(have, want) and ok(have_vision, want_vision):
        say(f"  PyTorch {have} и torchvision {have_vision} уже установлены")
        return
    index = TORCH_INDEX.format(tag=tag or "cpu")
    say(f"  PyTorch {want} и torchvision {want_vision} с {index} (~{'3' if tag else '0.3'} ГБ)")
    run([uv, "pip", "install", "--python", py, f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
         "--index-url", index], env=uv_env(home))


def lock_file(profile: str, tag: str | None) -> Path | None:
    """Exact verified versions for this platform, if we have them."""
    if profile != "full" or not tag or not IS_WIN:
        return None
    lock = SERVER_SRC / "requirements" / f"lock-win-{tag}.txt"
    return lock if lock.is_file() else None


def install_server(uv: Path, py: Path, home: Path, profile: str, dev: bool, lock: Path | None = None, tag: str | None = None) -> None:
    extras = "gpu,test" if profile == "full" else "test"
    target = f"{SERVER_SRC}[{extras}]"
    if lock is not None:
        say(f"  точные версии из {lock.name}")
        run([uv, "pip", "install", "--python", py, "-r", lock, "--extra-index-url", TORCH_INDEX.format(tag=tag),
             "--index-strategy", "unsafe-best-match"], env=uv_env(home))
    cmd = [uv, "pip", "install", "--python", py] + (["--no-deps"] if lock is not None else [])
    cmd += ["-e", target] if dev else [target]
    run(cmd, env=uv_env(home))
    if not dev:  # a plain install copies the code: refresh it even when the version did not change
        run([uv, "pip", "install", "--python", py, "--reinstall-package", "dronegen-server", "--no-deps", target],
            env=uv_env(home))


def site_packages(py: Path) -> Path:
    out = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return Path(out)


def rmtree(path: Path) -> None:
    """shutil.rmtree that also deletes read-only files (git objects on Windows)."""

    def retry(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def clear_dir(path: Path) -> None:
    """Delete what is inside ``path`` (the folder itself may be a shell's current directory)."""
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            rmtree(child)
        else:
            os.chmod(child, stat.S_IWRITE)
            child.unlink()


def install_vendor(py: Path, home: Path, name: str) -> None:
    """A pinned commit of a GitHub repo in <home>/vendor/<name>, importable from the venv."""
    repo, commit, paths = VENDOR[name]
    dest = home / "vendor" / name
    marker = dest / ".dronegen-commit"
    if marker.is_file() and marker.read_text().strip() == commit:
        say(f"  {name} {commit[:7]} уже на месте")
    else:
        if dest.exists():
            clear_dir(dest)
        dest.mkdir(parents=True, exist_ok=True)
        git = shutil.which("git")
        if git:  # only this commit, without the history; only the files in `paths`
            run([git, "init", "--quiet", dest])
            run([git, "-C", dest, "remote", "add", "origin", f"https://github.com/{repo}"])
            fetch = [git, "-C", dest, "fetch", "--quiet", "--depth", "1"]
            if paths:
                run([git, "-C", dest, "sparse-checkout", "set", "--no-cone", *paths])
                fetch.append("--filter=blob:none")
            run([*fetch, "origin", commit])
            run([git, "-C", dest, "checkout", "--quiet", "FETCH_HEAD"])
        else:  # no git: the commit as a zip from GitHub
            url = f"https://codeload.github.com/{repo}/zip/{commit}"
            say(f"  git не найден, скачиваю архив {url}")
            wanted = tuple(p.strip("/") for p in paths or ())
            with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=600).read())) as zf:
                for member in zf.infolist():
                    rel = member.filename.split("/", 1)[-1]  # without the "<repo>-<commit>/" folder
                    if not rel or member.is_dir() or (wanted and not rel.startswith(wanted)):
                        continue
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        marker.write_text(commit)
    # no installable package (TripoSR) or one with heavy pinned requirements (Hunyuan3D-2):
    # put the folder on sys.path of the venv instead of installing it
    pth = site_packages(py) / f"dronegen_{name.lower().replace('-', '_')}.pth"
    pth.write_text(str(dest) + "\n", encoding="utf-8")
    say(f"  {name} подключён: {dest}")


# ---------------------------------------------------------------------------- files
START_BAT = """@echo off
rem DroneGen server. Close this window (or press Ctrl+C) to stop it.
chcp 65001 >nul
title DroneGen server
"%~dp0venv\\Scripts\\python.exe" -m dronegen_server --home "%~dp0." %*
if errorlevel 1 pause
"""

STUDIO_BAT = """@echo off
rem DroneGen Studio: starts the server and opens the chat window. Close this window to stop the server.
chcp 65001 >nul
title DroneGen server
"%~dp0venv\\Scripts\\python.exe" -m dronegen_server --home "%~dp0." --open-studio %*
if errorlevel 1 pause
"""

START_SH = """#!/bin/sh
# DroneGen server. Ctrl+C stops it. With --open-studio the chat window opens too.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/venv/bin/python" -m dronegen_server --home "$DIR" "$@"
"""


def write_start_scripts(home: Path) -> None:
    if IS_WIN:
        (home / "start_server.bat").write_text(START_BAT, encoding="utf-8")
        (home / "start_studio.bat").write_text(STUDIO_BAT, encoding="utf-8")
    else:
        sh = home / "start_server.sh"
        sh.write_text(START_SH, encoding="utf-8")
        sh.chmod(0o755)


def write_config(home: Path, profile: str, gpu: dict | None, port: int) -> None:
    path = home / "config.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.setdefault("host", "127.0.0.1")
    data.setdefault("port", port)
    data.setdefault("device", "auto")
    data.setdefault("offline", True)
    data.setdefault("idle_unload_s", 600)
    data.setdefault("zimage_gguf", "z_image_turbo-Q4_K_M.gguf")
    data["profile"] = profile
    data["gpu"] = gpu["name"] if gpu else None
    data["installed"] = datetime.datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def install_addon(blenders: list[tuple[Path, tuple]], home: Path, port: int, build_only: bool = False) -> list[str]:
    """Build the extension with Blender's own tool (it validates the manifest) and install it."""
    done = []
    if not blenders:
        say("  Blender 4.2+ не найден — аддон не установлен. Укажите путь: --blender \"C:\\...\\blender.exe\"")
        return done
    home = home.resolve()
    stage = Path(tempfile.mkdtemp(prefix="dronegen-addon-"))
    src = stage / "dronegen"
    shutil.copytree(ADDON_SRC, src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    info = {"home": str(home), "server_url": f"http://127.0.0.1:{port}"}
    (src / "install_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = home / "addon"
    out_dir.mkdir(parents=True, exist_ok=True)
    for exe, version in blenders:
        if version < (4, 2, 0):
            say(f"  пропускаю Blender {'.'.join(map(str, version))}: нужен 4.2 или новее")
            continue
        for old in out_dir.glob("dronegen-*.zip"):
            old.unlink()
        run([exe, "--factory-startup", "-b", "--command", "extension", "build",
             "--source-dir", src, "--output-dir", out_dir], capture=True)
        zips = sorted(out_dir.glob("dronegen-*.zip"))
        if not zips:
            raise InstallError("Blender не собрал пакет аддона")
        if build_only:
            say(f"  пакет аддона: {zips[-1]}  (Blender: Edit → Preferences → Get Extensions → Install from Disk…)")
            done.append(str(zips[-1]))
            break
        run([exe, "-b", "--command", "extension", "install-file", "-r", "user_default", "-e", zips[-1]], capture=True)
        label = "Blender " + ".".join(map(str, version))
        say(f"  аддон DroneGen установлен и включён: {label}")
        done.append(label)
    shutil.rmtree(stage, ignore_errors=True)
    return done


def blender_running() -> bool:
    try:
        if IS_WIN:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq blender.exe"], capture_output=True, text=True).stdout
            return "blender.exe" in out.lower()
        return subprocess.run(["pgrep", "-x", "blender"], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    global _log_file
    p = argparse.ArgumentParser(description="Установка DroneGen (сервер, модели, аддон Blender)")
    p.add_argument("--home", type=Path, default=default_home(), help="папка установки (по умолчанию %(default)s)")
    p.add_argument("--profile", choices=["auto", "full", "light"], default="auto",
                   help="full: нейросети на GPU; light: без PyTorch (силуэт и тест); auto: full, если есть NVIDIA")
    p.add_argument("--cuda", default="auto", help="сборка PyTorch: auto, cu130, cu126 или cpu")
    p.add_argument("--skip-sdxl", action="store_true", help="не скачивать запасную модель SDXL (~7 ГБ)")
    p.add_argument("--skip-llm", action="store_true", help="не скачивать локальную модель чата (~6 ГБ)")
    p.add_argument("--skip-models", action="store_true", help="не скачивать веса (только программы)")
    p.add_argument("--no-addon", action="store_true", help="не ставить аддон в Blender")
    p.add_argument("--addon-zip", action="store_true", help="только собрать пакет аддона (runtime/addon), не устанавливая")
    p.add_argument("--no-selfcheck", action="store_true")
    p.add_argument("--blender", action="append", default=[], help="путь к blender(.exe), можно несколько раз")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--dev", action="store_true", help="код сервера из этой папки (editable), для разработки")
    p.add_argument("--unlocked", action="store_true", help="свежие версии пакетов вместо проверенных (lock-файла)")
    args = p.parse_args(argv)

    # Russian text must survive pipes and log files (the console code page is not UTF-8)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    home: Path = args.home.resolve()
    home.mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(exist_ok=True)
    _log_file = open(home / "logs" / "install.log", "a", encoding="utf-8")
    say(f"=== DroneGen install {datetime.datetime.now():%Y-%m-%d %H:%M} -> {home}")

    total = 9
    try:
        step(1, total, "Проверка компьютера")
        gpu = detect_gpu()
        mem = detect_memory()
        free_gb = shutil.disk_usage(home).free / 1024**3
        say(f"  ОС: {platform.system()} {platform.release()} ({platform.machine()})")
        if gpu:
            say(f"  GPU: {gpu['name']}, {gpu['vram_gb']} ГБ VRAM (занято {gpu['vram_used_gb']} ГБ), "
                f"драйвер {gpu['driver']}, CUDA {gpu['cuda'][0]}.{gpu['cuda'][1]}")
        else:
            say("  GPU NVIDIA не найден")
        line = f"  RAM: {mem['total_gb']:.0f} ГБ, свободно {mem['available_gb']:.1f} ГБ"
        if "commit_limit_gb" in mem:
            line += f"; лимит выделения (RAM + файл подкачки) {mem['commit_limit_gb']:.0f} ГБ"
            line += ", файл подкачки по выбору системы" if mem["pagefile_auto"] else ", файл подкачки фиксированный"
        say(line)
        say(f"  Диск: свободно {free_gb:.0f} ГБ в {home}")
        profile = args.profile if args.profile != "auto" else ("full" if gpu else "light")
        tag = pick_cuda_build(gpu, args.cuda) if profile == "full" else None
        say(f"  Профиль: {profile}" + (f", PyTorch {TORCH_VERSION}+{tag}" if tag else
                                        (", PyTorch для CPU (медленно)" if profile == "full" else ", без PyTorch")))
        if gpu and gpu["vram_gb"] < 6 and profile == "full":
            say("  ! VRAM меньше 6 ГБ: Z-Image и TripoSR пойдут в медленном режиме с выгрузкой слоёв")
        if gpu and gpu["vram_used_gb"] > 2.5:
            say(f"  ! Сейчас занято {gpu['vram_used_gb']} ГБ видеопамяти другими программами — "
                "перед генерацией закройте лишние (например, EasyDiffusion)")
        if not mem.get("pagefile_auto", True) and mem.get("commit_limit_gb", 99) < mem["total_gb"] + 8 and profile == "full":
            say("  ! Файл подкачки маленький: при загрузке больших моделей возможна ошибка WinError 1455. "
                "Рекомендуется «Размер по выбору системы»")
        need = (45 - (7 if args.skip_sdxl else 0) - (6 if args.skip_llm else 0)) if profile == "full" else 2
        if free_gb < need and not args.skip_models:
            raise InstallError(f"Нужно ~{need} ГБ свободного места, есть {free_gb:.0f} ГБ")

        step(2, total, "uv и Python " + PYTHON_VERSION)
        uv = ensure_uv(home)
        py = ensure_venv(uv, home)

        step(3, total, "PyTorch")
        if profile == "full":
            install_torch(uv, py, home, tag)
        else:
            say("  не нужен (лёгкий профиль)")

        step(4, total, "Сервер DroneGen и зависимости")
        install_server(uv, py, home, profile, args.dev, None if args.unlocked else lock_file(profile, tag), tag)
        if profile == "full" and tag:
            have = torch_state(py)
            if not have.endswith("+" + tag):  # a dependency pulled the CPU build from PyPI
                say(f"  PyTorch заменился на {have}, возвращаю сборку {tag}")
                install_torch(uv, py, home, tag)

        step(5, total, "Код моделей: TripoSR, Hunyuan3D-2")
        if profile == "full":
            for name in VENDOR:
                install_vendor(py, home, name)
        else:
            say("  не нужен (лёгкий профиль)")

        step(6, total, "Веса моделей")
        if args.skip_models:
            say("  пропущено (--skip-models)")
        else:
            cmd = [py, "-m", "dronegen_server.download", "--home", home, "--profile", profile]
            if args.skip_sdxl:
                cmd.append("--skip-sdxl")
            if args.skip_llm:
                cmd.append("--skip-llm")
            run(cmd)

        step(7, total, "Настройки и скрипт запуска")
        write_config(home, profile, gpu, args.port)
        write_start_scripts(home)
        if IS_WIN:  # «DroneGen Studio» on the desktop and in the Start menu: a click, no console window
            run([py, "-m", "dronegen_server", "--home", home, "--make-shortcuts"])
        say(f"  {home / 'config.json'}")
        say(f"  {home / ('start_server.bat' if IS_WIN else 'start_server.sh')}")

        step(8, total, "Самопроверка (офлайн)")
        if args.no_selfcheck:
            say("  пропущено")
        else:
            run([py, "-m", "dronegen_server.selfcheck", "--home", home, "--offline"])

        step(9, total, "Аддон Blender")
        installed = []
        if args.no_addon:
            say("  пропущено (--no-addon)")
        else:
            installed = install_addon(find_blenders(args.blender), home, args.port, build_only=args.addon_zip)
            if installed and not args.addon_zip and blender_running():
                say("  ! Blender сейчас открыт: перезапустите его. Если вкладки DroneGen нет — "
                    "Edit → Preferences → Add-ons → включите DroneGen")
    except InstallError as e:
        say(f"\nОШИБКА: {e}\nЛог: {home / 'logs' / 'install.log'}. Исправьте и запустите установку ещё раз.")
        return 1
    except KeyboardInterrupt:
        say("\nПрервано. Повторный запуск продолжит с того же места.")
        return 130

    start = home / ("start_server.bat" if IS_WIN else "start_server.sh")
    say("\nГотово.")
    say(f"  Запуск сервера: {start}   (или кнопка «Запустить сервер» в Blender)")
    say("  Blender: 3D-вид → N → вкладка DroneGen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
