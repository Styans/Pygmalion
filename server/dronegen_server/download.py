"""Download model weights (used by install.py; safe to re-run, finished ones are skipped).

    python -m dronegen_server.download                  # everything for the full profile
    python -m dronegen_server.download --skip-sdxl --skip-llm
    python -m dronegen_server.download --only triposr,dino
    python -m dronegen_server.download --profile light  # only the background-removal model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import weights
from .config import Settings, utf8_stdio


def keys_for(profile: str, skip_sdxl: bool, skip_llm: bool = False) -> list[str]:
    if profile == "light":
        return ["u2net"]
    keys = ["u2net", "zimage_gguf", "zimage", "hunyuan_mini", "hunyuan_vae", "hunyuan_mv", "triposr", "dino"]
    if not skip_llm:  # the local chat model: llama.cpp + Qwen3-VL-8B (~6 GB)
        keys += ["llamacpp", "qwen3vl"]
    if not skip_sdxl:  # fallbacks last
        keys += ["sdxl_vae", "sdxl"]
    return keys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Скачать веса моделей DroneGen")
    p.add_argument("--home", type=Path, default=None, help="папка установки DroneGen")
    p.add_argument("--models-dir", type=Path, default=None, help="папка для весов (по умолчанию <home>/models)")
    p.add_argument("--profile", choices=["full", "light"], default="full")
    p.add_argument("--skip-sdxl", action="store_true")
    p.add_argument("--skip-llm", action="store_true", help="без локальной модели чата (~6 ГБ): тогда чат только через облако")
    p.add_argument("--only", default="", help="через запятую: " + ", ".join(weights.WEIGHTS))
    args = p.parse_args(argv)
    utf8_stdio()

    settings = Settings.load(args.home, models_dir=args.models_dir, offline=False)
    settings.ensure_dirs()
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or keys_for(args.profile, args.skip_sdxl, args.skip_llm)
    unknown = [k for k in keys if k not in weights.WEIGHTS]
    if unknown:
        p.error("неизвестные веса: " + ", ".join(unknown))
    total = sum(weights.WEIGHTS[k].size_gb for k in keys if not weights.present(settings, k))
    print(f"Папка весов: {settings.models}; осталось скачать ~{total:.1f} ГБ", flush=True)
    try:
        weights.download(settings, keys, log=lambda m: print(m, flush=True))
    except KeyboardInterrupt:
        print("Прервано. Повторный запуск продолжит загрузку.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 - report any network/disk problem in one line
        print(f"Ошибка загрузки: {e}\nПроверьте интернет и запустите ещё раз — загрузка продолжится.", file=sys.stderr)
        return 1
    print("Все веса на месте.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
