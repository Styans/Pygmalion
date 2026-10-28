"""Server settings. Priority: command-line flags > ``<home>/config.json`` > defaults.

``home`` is the install folder (``runtime/`` in the project folder): it holds the venv,
``models/``, ``outputs/`` and ``logs/``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188


def utf8_stdio() -> None:
    """Russian text and arrows must survive pipes and log files on Windows (cp1251/cp866 consoles)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def default_home() -> Path:
    """DRONEGEN_HOME, else the folder that holds the venv we run from (``<home>/venv``)."""
    env = os.environ.get("DRONEGEN_HOME")
    if env:
        return Path(env)
    prefix = Path(sys.prefix).resolve()
    if prefix.name == "venv":
        return prefix.parent
    return Path.cwd() / "runtime"


@dataclass
class Settings:
    home: Path = field(default_factory=default_home)
    models_dir: Path | None = None  # default: <home>/models
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = "auto"  # auto | cuda | cpu
    backends: list[str] | None = None  # None: every registered backend
    offline: bool = True  # never touch the network for weights (they are downloaded by install.py)
    token: str = ""  # optional Bearer token, for running the server on another machine
    idle_unload_s: float = 600.0  # unload GPU models after this long without jobs (0: never)
    max_jobs_kept: int = 300
    max_upload_mb: int = 25
    zimage_gguf: str = "z_image_turbo-Q4_K_M.gguf"
    zimage_keep_text_encoder: bool = False  # True: keep Qwen3-4B (~8 GB RAM) loaded between prompts
    log_level: str = "info"

    @property
    def models(self) -> Path:
        return Path(self.models_dir) if self.models_dir else self.home / "models"

    @property
    def outputs(self) -> Path:
        return self.home / "outputs"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.models, self.outputs, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def backend_enabled(self, name: str) -> bool:
        return self.backends is None or name in self.backends

    @classmethod
    def load(cls, home: Path | None = None, **overrides) -> "Settings":
        """Settings for ``home`` from its config.json, then ``overrides`` (None values are ignored)."""
        home = (Path(home) if home else default_home()).resolve()  # external programs run in other folders
        data: dict = {}
        cfg = home / "config.json"
        if cfg.is_file():
            data = json.loads(cfg.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        data.update({k: v for k, v in overrides.items() if v is not None})
        data["home"] = home
        if data.get("models_dir"):
            data["models_dir"] = Path(data["models_dir"]).resolve()
        return cls(**data)

    def save(self) -> Path:
        """Write the persistent part of the settings to ``<home>/config.json``."""
        data = asdict(self)
        data.pop("home")
        data["models_dir"] = str(self.models_dir) if self.models_dir else None
        path = self.home / "config.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def apply_env(self) -> None:
        """Point Hugging Face and rembg at our model folder. Call before importing them."""
        os.environ["HF_HOME"] = str(self.models / "hf")
        os.environ["U2NET_HOME"] = str(self.models / "u2net")
        os.environ["HY3DGEN_MODELS"] = str(self.models / "hy3dgen")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
