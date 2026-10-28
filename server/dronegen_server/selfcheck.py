"""Post-install check:  python -m dronegen_server.selfcheck [--home DIR] [--offline] [--models]

* imports, PyTorch/CUDA, marching cubes library, which weights are downloaded;
* a test picture -> silhouette mesh through the real job queue;
* an exact drawing (parametric) and the Studio's database;
* ``--offline``: connections to anything but this computer are blocked while checking,
  so a hidden download would fail here instead of at a show;
* ``--models``: also load every available neural backend and run a small job, and ask the local
  chat model a question.
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import time
from pathlib import Path

from . import __version__
from .config import Settings, utf8_stdio


def block_network() -> None:
    """Allow only loopback connections in this process."""

    def loopback(address) -> bool:
        host = address[0] if isinstance(address, tuple) else address
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host in ("localhost",)

    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not loopback(address):
            raise OSError(f"selfcheck --offline: сетевое подключение к {address} запрещено")
        return original_connect(self, address)

    socket.socket.connect = guarded_connect

    original_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in (None, "localhost", "127.0.0.1", "::1"):
            raise OSError(f"selfcheck --offline: DNS-запрос {host} запрещён")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.getaddrinfo = guarded_getaddrinfo


def wait(manager, job, timeout: float = 900.0):
    t0 = time.time()
    while job.status not in ("done", "error", "cancelled"):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"задача {job.backend} не завершилась за {timeout:.0f} с")
        time.sleep(0.1)
    return job


def check_studio_store() -> list[str]:
    import tempfile

    from .store import Store

    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            chat = store.add_chat(store.add_project("проверка")["id"])
            store.add_message(chat["id"], {"role": "user", "content": [{"type": "text", "text": "привет"}]})
            assert store.messages(chat["id"])[0]["content"][0]["text"] == "привет"
            store.close()
        print("База студии (SQLite): работает")
        return []
    except Exception as e:  # noqa: BLE001
        return [f"база студии: {e}"]


def check_chat_model(settings, manager) -> list[str]:
    """Start the local chat model, ask one question, stop it (it needs the GPU the other models used)."""
    import asyncio

    from .llm import Providers
    from .store import StudioConfig

    config = StudioConfig(settings.home / "studio")
    providers = Providers(settings, config, manager)
    ok, reason = providers.local.available()
    if not ok:
        print(f"Локальная модель чата: пропущено — {reason}")
        return []
    c_ok, c_reason = providers.claude.available()
    print("Облачный Claude: " + ("ключ задан (используется, только если выбран режим «Облако»)" if c_ok else "не используется"))
    try:
        t0 = time.time()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Ответь одним словом: сколько будет два плюс два?"}]}]
        answer = asyncio.run(providers.local.text("Отвечай по-русски, коротко.", messages))
        print(f"Локальная модель чата (Qwen3-VL): ответила за {time.time() - t0:.1f} с (с запуском): {answer.strip()[:60]}")
        return [] if answer.strip() else ["локальная модель чата ответила пустотой"]
    except Exception as e:  # noqa: BLE001
        return [f"локальная модель чата: {e}"]
    finally:
        asyncio.run(providers.close())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Самопроверка сервера DroneGen")
    p.add_argument("--home", type=Path, default=None)
    p.add_argument("--offline", action="store_true", help="запретить сеть на время проверки")
    p.add_argument("--models", action="store_true", help="загрузить нейросети и сделать по маленькой задаче")
    args = p.parse_args(argv)
    utf8_stdio()

    settings = Settings.load(args.home, offline=True, idle_unload_s=0)
    settings.ensure_dirs()
    settings.apply_env()
    if args.offline:
        block_network()

    from . import imaging, memory, meshing, weights
    from .backends import Registry
    from .jobs import JobManager

    problems: list[str] = []
    print(f"DroneGen server {__version__}, Python {sys.version.split()[0]}, данные: {settings.home}")
    gpu = memory.gpu_info()
    if gpu.get("available"):
        print(f"GPU: {gpu['name']}, свободно {gpu['free_gb']} из {gpu['total_gb']} ГБ (torch {gpu['torch']}, CUDA {gpu['cuda']})")
    else:
        print(f"GPU: нет — {gpu.get('reason')}")
    print(f"Marching cubes: {', '.join(meshing.mc_backends())}; rembg: {'да' if imaging.rembg_available() else 'нет'}")
    for key, spec in weights.WEIGHTS.items():
        print(f"  веса {'+' if weights.present(settings, key) else '-'} {spec.title}")

    registry = Registry(settings)
    manager = JobManager(settings, registry)
    manager.start()
    try:
        img_job = wait(manager, manager.submit("txt2img", "mock", {"prompt": "selfcheck", "seed": 1, "steps": 2}))
        if img_job.status != "done":
            problems.append(f"тестовая картинка: {img_job.error}")
        else:
            from PIL import Image

            picture = Image.open(settings.outputs / img_job.files["image"])
            picture.load()
            mesh_job = wait(manager, manager.submit("img2mesh", "silhouette", {"remove_bg": False}, {"image": picture}))
            if mesh_job.status != "done":
                problems.append(f"силуэт → 3D: {mesh_job.error}")
            else:
                r = mesh_job.result
                print(f"Силуэт → 3D: {r['faces']} треугольников, замкнутая: {r['watertight']}, "
                      f"{mesh_job.metrics.get('seconds', 0):.1f} с")
                if not r["watertight"]:
                    problems.append("силуэт → 3D: сетка не замкнута")
            build_job = wait(manager, manager.submit("build", "parametric", {"template": "dombra"}))
            if build_job.status != "done" or not build_job.result.get("watertight"):
                problems.append(f"точный чертёж: {build_job.error or 'сетка не замкнута'}")
            else:
                print(f"Точный чертёж «домбра»: {build_job.result['faces']} треугольников, "
                      f"{build_job.metrics.get('seconds', 0):.2f} с")
            if weights.present(settings, "u2net"):
                bg_job = wait(manager, manager.submit("img2mesh", "silhouette", {"bg_method": "rembg"}, {"image": picture}))
                if bg_job.status != "done":
                    problems.append(f"удаление фона rembg: {bg_job.error}")
                else:
                    print("Удаление фона rembg: работает")
            if args.models:
                for backend in registry.all():
                    if not backend.heavy:
                        continue
                    ok, reason = backend.check()
                    if not ok:
                        print(f"{backend.title}: пропущено — {reason}")
                        continue
                    if backend.kind == "txt2img":
                        params = {"prompt": "a red star, white background", "width": 512, "height": 512, "steps": 4}
                        job = manager.submit("txt2img", backend.name, params)
                    else:
                        inputs = {"image": picture}
                        if backend.name == "hunyuan_mv":  # needs several views of one object
                            cutout = imaging.remove_background(picture, "auto")[0]
                            inputs["views"] = {"front": cutout, "left": cutout}
                        job = manager.submit("img2mesh", backend.name, {"mc_resolution": 128}, inputs)
                    wait(manager, job, timeout=1800)
                    if job.status == "done":
                        print(f"{backend.title}: работает, {job.metrics.get('total_seconds', 0):.1f} с "
                              f"(загрузка {job.metrics.get('load_seconds', 0):.1f} с)")
                    else:
                        problems.append(f"{backend.title}: {job.error}")
                    manager.unload_all()
        if args.models:
            problems += check_chat_model(settings, manager)
    finally:
        manager.stop()
    problems += check_studio_store()
    here = Path(__file__).parent
    missing = [n for n in ("studio/index.html", "studio/app.js", "agents/knowledge/dombra.md") if not (here / n).is_file()]
    if missing:  # a packaging mistake: the chat window or the knowledge sheets would be missing
        problems.append("в установленном пакете нет файлов: " + ", ".join(missing))

    if problems:
        print("\nПроблемы:\n  - " + "\n  - ".join(problems))
        return 1
    print("\nСамопроверка пройдена.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
