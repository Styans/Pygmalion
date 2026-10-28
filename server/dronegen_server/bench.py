"""Measure a backend on this machine: load time, job time, peak VRAM and RAM.

    python -m dronegen_server.bench txt2img zimage_turbo --size 768 --runs 2
    python -m dronegen_server.bench img2mesh triposr --image picture.png --mc 256 --chunk 8192

Every run goes through the real job queue. Results are printed as Markdown table rows and
appended to ``<home>/logs/bench.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from .config import Settings, utf8_stdio


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Замеры бэкенда DroneGen")
    p.add_argument("kind", choices=["txt2img", "img2mesh"])
    p.add_argument("backend")
    p.add_argument("--home", type=Path, default=None)
    p.add_argument("--runs", type=int, default=2, help="первый прогон включает загрузку модели")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--prompt", default="golden eagle with wings fully spread, front view, single object, centered, "
                                        "plain white background, bold simple silhouette, flat colors")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--mc", type=int, default=None, help="mc_resolution")
    p.add_argument("--chunk", type=int, default=None, help="chunk_size (TripoSR)")
    p.add_argument("--no-remove-bg", action="store_true")
    p.add_argument("--mode", default=None, choices=["offload", "group", "sequential"], help="режим памяти (txt2img)")
    p.add_argument("--note", default="")
    args = p.parse_args(argv)
    utf8_stdio()

    settings = Settings.load(args.home, idle_unload_s=0)
    settings.ensure_dirs()
    settings.apply_env()

    from PIL import Image

    from . import memory
    from .backends import Registry
    from .jobs import JobManager

    registry = Registry(settings)
    manager = JobManager(settings, registry)
    backend = registry.get(args.kind, args.backend)
    ok, reason = backend.check()
    if not ok:
        print(f"{backend.title}: недоступен — {reason}")
        return 1
    manager.start()
    gpu = memory.gpu_info()
    rows = []
    try:
        for run in range(args.runs):
            free_before = memory.vram_free_gb()
            if args.kind == "txt2img":
                params = {"prompt": args.prompt, "width": args.size, "height": args.size, "seed": args.seed + run}
                if args.steps:
                    params["steps"] = args.steps
                if args.mode:
                    params["memory_mode"] = args.mode
                job = manager.submit("txt2img", args.backend, params)
            else:
                if args.image is None:
                    p.error("для img2mesh нужен --image")
                img = Image.open(args.image)
                img.load()
                params = {"remove_bg": not args.no_remove_bg, "mc_resolution": args.mc, "chunk_size": args.chunk}
                job = manager.submit("img2mesh", args.backend, params, {"image": img})
            while job.status not in ("done", "error", "cancelled"):
                time.sleep(0.1)
            m = job.metrics
            row = {
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "kind": args.kind,
                "backend": args.backend,
                "run": run + 1,
                "status": job.status,
                "error": job.error,
                "params": {k: v for k, v in job.params.items() if k not in ("prompt", "negative")},
                "load_s": m.get("load_seconds"),
                "job_s": m.get("seconds"),
                "total_s": m.get("total_seconds"),
                "vram_peak_torch_gb": m.get("vram_peak_torch_gb"),
                "vram_device_peak_gb": m.get("vram_device_peak_gb"),
                "vram_free_before_gb": round(free_before, 2),
                "ram_peak_gb": m.get("ram_peak_gb"),
                "result": job.result,
                "warnings": job.warnings,
                "files": job.files,
                "gpu": gpu.get("name"),
                "host": platform.node(),
                "note": args.note,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    finally:
        manager.stop()
    with open(settings.logs / "bench.jsonl", "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n| прогон | статус | загрузка, с | задача, с | пик VRAM (torch), ГБ | пик VRAM (устройство), ГБ | RAM, ГБ |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['status']} | {r['load_s'] or '—'} | {r['job_s']} | {r['vram_peak_torch_gb']} | "
              f"{r['vram_device_peak_gb']} | {r['ram_peak_gb']} |")
    return 0 if all(r["status"] == "done" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
