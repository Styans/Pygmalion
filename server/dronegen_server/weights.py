"""Which weights each backend needs, where they live, and whether they are downloaded.

Repositories are downloaded with ``allow_patterns`` into plain folders under ``<models>/``
(no symlinks, works offline by path). A marker in ``<models>/.complete/`` is written only
after a download finished, so a half-downloaded folder never counts as present.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass(frozen=True)
class WeightSpec:
    key: str
    title: str
    repo: str  # Hugging Face repo id ("" for rembg's own download)
    folder: str  # folder under <models>; "" -> the Hugging Face cache (HF_HOME)
    patterns: tuple[str, ...]
    size_gb: float
    group: str  # what needs it: zimage | sdxl | triposr | rembg


WEIGHTS: dict[str, WeightSpec] = {
    s.key: s
    for s in (
        WeightSpec(
            "zimage",
            "Z-Image-Turbo: текстовый энкодер Qwen3-4B, VAE, токенизатор",
            "Tongyi-MAI/Z-Image-Turbo",
            "zimage",
            (
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "tokenizer/*",
                "vae/*",
                "transformer/config.json",
            ),
            7.9,
            "zimage",
        ),
        WeightSpec(
            "zimage_gguf",
            "Z-Image-Turbo: transformer GGUF Q4_K_M",
            "jayn7/Z-Image-Turbo-GGUF",
            "zimage-gguf",
            ("z_image_turbo-Q4_K_M.gguf",),
            4.64,
            "zimage",
        ),
        WeightSpec(
            "sdxl",
            "SDXL base 1.0 (fp16)",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "sdxl",
            (
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors",
                "text_encoder_2/config.json",
                "text_encoder_2/model.fp16.safetensors",
                "tokenizer/*",
                "tokenizer_2/*",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json",
            ),
            6.8,
            "sdxl",
        ),
        WeightSpec(
            "sdxl_vae",
            "SDXL VAE fp16-fix",
            "madebyollin/sdxl-vae-fp16-fix",
            "sdxl-vae-fp16-fix",
            ("config.json", "diffusion_pytorch_model.safetensors"),
            0.34,
            "sdxl",
        ),
        WeightSpec(
            "triposr", "TripoSR", "stabilityai/TripoSR", "triposr", ("config.yaml", "model.ckpt"), 1.68, "triposr"
        ),
        # TripoSR builds its image encoder from this config with hf_hub_download -> keep it in HF_HOME
        WeightSpec("dino", "DINO ViT-B/16 config", "facebook/dino-vitb16", "", ("config.json",), 0.0, "triposr"),
        WeightSpec("u2net", "rembg u2net (удаление фона)", "", "u2net", ("u2net.onnx",), 0.18, "rembg"),
    )
}

# hy3dgen looks for <HY3DGEN_MODELS>/<repo id>/<subfolder>: keep exactly that layout
_HY = "hy3dgen/tencent/"
WEIGHTS.update(
    {
        s.key: s
        for s in (
            WeightSpec(
                "hunyuan_mini",
                "Hunyuan3D-2mini-turbo (форма по одной картинке)",
                "tencent/Hunyuan3D-2mini",
                _HY + "Hunyuan3D-2mini",
                (
                    "hunyuan3d-dit-v2-mini-turbo/config.yaml",
                    "hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors",
                    "hunyuan3d-vae-v2-mini-turbo/config.yaml",
                    "hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors",
                ),
                3.94,
                "hunyuan",
            ),
            WeightSpec(
                "hunyuan_mv",
                "Hunyuan3D-2mv-turbo (форма по нескольким видам)",
                "tencent/Hunyuan3D-2mv",
                _HY + "Hunyuan3D-2mv",
                ("hunyuan3d-dit-v2-mv-turbo/config.yaml", "hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors"),
                4.59,
                "hunyuan",
            ),
            WeightSpec(
                "hunyuan_vae",
                "Hunyuan3D-2: быстрый декодер формы (FlashVDM)",
                "tencent/Hunyuan3D-2",
                _HY + "Hunyuan3D-2",
                ("hunyuan3d-vae-v2-0-turbo/config.yaml", "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors"),
                0.39,
                "hunyuan",
            ),
            WeightSpec(
                "qwen3vl",
                "Qwen3-VL-8B (локальный ИИ чата, видит картинки)",
                "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                "llm/qwen3-vl-8b",
                ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"),
                5.38,
                "llm",
            ),
            WeightSpec("llamacpp", "llama.cpp (сервер локального ИИ)", "", "llamacpp", (), 0.61, "llm"),
        )
    }
)

GROUPS = {
    "zimage": ("zimage", "zimage_gguf"),
    "sdxl": ("sdxl", "sdxl_vae"),
    "triposr": ("triposr", "dino"),
    "rembg": ("u2net",),
    "hunyuan_mini": ("hunyuan_mini",),
    "hunyuan_mv": ("hunyuan_mv", "hunyuan_vae"),
    "llm": ("qwen3vl", "llamacpp"),
}

# llama.cpp builds from GitHub. CUDA 12.4: the CUDA 13.x builds need a newer driver than 591.x.
LLAMACPP_TAG = "b11319"
LLAMACPP_ASSETS = {
    "win32": (f"llama-{LLAMACPP_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"),
    "linux": (f"llama-{LLAMACPP_TAG}-bin-ubuntu-vulkan-x64.zip",),
}


def folder(settings: Settings, key: str) -> Path:
    spec = WEIGHTS[key]
    return settings.models / spec.folder if spec.folder else settings.models / "hf"


def _marker(settings: Settings, key: str) -> Path:
    return settings.models / ".complete" / f"{key}.json"


def present(settings: Settings, key: str) -> bool:
    return _marker(settings, key).is_file()


def missing(settings: Settings, group: str) -> list[WeightSpec]:
    return [WEIGHTS[k] for k in GROUPS[group] if not present(settings, k)]


def missing_message(specs: list[WeightSpec]) -> str:
    names = ", ".join(s.title for s in specs)
    return f"не скачаны веса: {names}. Запустите установку: python install.py"


def mark_complete(settings: Settings, key: str, files: list[str]) -> None:
    m = _marker(settings, key)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps({"key": key, "files": files, "time": time.time()}, ensure_ascii=False), encoding="utf-8")


def download(settings: Settings, keys: list[str], log=print) -> None:
    """Download the given weights (network required). Already complete ones are skipped."""
    settings.apply_env()  # HF_HOME / U2NET_HOME; offline must be False here
    from huggingface_hub import hf_hub_download, snapshot_download

    for key in keys:
        spec = WEIGHTS[key]
        if present(settings, key):
            log(f"[есть] {spec.title}")
            continue
        log(f"[загрузка] {spec.title} (~{spec.size_gb:.2f} ГБ)")
        started = time.time()
        if key == "u2net":
            files = [_download_u2net(settings)]
        elif key == "llamacpp":
            files = _download_llamacpp(settings, log)
        elif spec.folder:
            target = folder(settings, key)
            snapshot_download(spec.repo, allow_patterns=list(spec.patterns), local_dir=target)
            files = sorted(
                p.relative_to(target).as_posix()
                for p in target.rglob("*")
                if p.is_file() and ".cache" not in p.relative_to(target).parts
            )
            if not files:
                raise RuntimeError(f"{spec.repo}: ничего не скачалось по шаблонам {spec.patterns}")
        else:
            files = [hf_hub_download(spec.repo, f) for f in spec.patterns]
        mark_complete(settings, key, files)
        log(f"[готово] {spec.title} за {time.time() - started:.0f} с")


def llama_server(settings: Settings) -> Path | None:
    """The llama-server executable, if the llama.cpp build is installed."""
    root = folder(settings, "llamacpp")
    for name in ("llama-server.exe", "llama-server"):
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    return None


def _download_llamacpp(settings: Settings, log) -> list[str]:
    import sys
    import urllib.request
    import zipfile

    assets = LLAMACPP_ASSETS.get("win32" if sys.platform == "win32" else "linux")
    if assets is None:
        raise RuntimeError("для этой ОС нет готовой сборки llama.cpp")
    root = folder(settings, "llamacpp")
    root.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TAG}/{asset}"
        part = root / (asset + ".part")
        log(f"  {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(part) as zf:
            zf.extractall(root)
        part.unlink()
    exe = llama_server(settings)
    if exe is None:
        raise RuntimeError(f"в архивах llama.cpp нет llama-server ({root})")
    return [exe.relative_to(root).as_posix()]


def _download_u2net(settings: Settings) -> str:
    """rembg fetches its ONNX model itself (into U2NET_HOME) on the first session."""
    from . import imaging

    imaging.rembg_session("u2net")
    root = folder(settings, "u2net")
    # rembg >= 2.0.85 keeps models in <U2NET_HOME>/models/<name>/, older versions flat in U2NET_HOME
    found = sorted(root.rglob("u2net.onnx"))
    if not found:
        raise RuntimeError(f"rembg не скачал модель в {root}")
    return found[0].relative_to(root).as_posix()
