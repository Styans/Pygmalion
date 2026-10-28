"""Common interface of all backends and their registry.

A backend is one way to do one kind of job (``txt2img`` or ``img2mesh``). Adding a new one
(or a new kind, e.g. a future ``mesh2points``) is a subclass with ``@register``; the job
queue, the HTTP API and the Blender add-on pick it up by name.

Module imports must stay light: heavy libraries (torch, diffusers) are imported in
``load()`` / ``run()``, so the server starts without them.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..errors import UserError

if TYPE_CHECKING:
    from ..config import Settings
    from ..jobs import JobContext


class Backend:
    kind: str = ""  # txt2img | img2mesh
    name: str = ""
    title: str = ""
    needs_gpu: bool = False
    heavy: bool = False  # big weights: only one heavy backend stays loaded at a time
    defaults: dict = {}

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.loaded = False

    # -- availability -------------------------------------------------------------------
    def check(self) -> tuple[bool, str]:
        """(usable, reason why not). Must be quick: called by /health."""
        return True, ""

    def describe(self) -> dict:
        ok, reason = self.check()
        return {
            "name": self.name,
            "title": self.title,
            "available": ok,
            "reason": reason,
            "needs_gpu": self.needs_gpu,
            "loaded": self.loaded,
            "defaults": dict(self.defaults),
        }

    # -- lifecycle ----------------------------------------------------------------------
    def validate(self, params: dict) -> dict:
        """Fill defaults and check values; raise UserError with a readable message."""
        out = dict(self.defaults)
        out.update({k: v for k, v in params.items() if v is not None})
        return out

    def load(self, ctx: "JobContext") -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def run(self, ctx: "JobContext", params: dict, inputs: dict) -> dict:
        raise NotImplementedError


class Txt2ImgBackend(Backend):
    kind = "txt2img"
    defaults = {"negative": "", "width": 768, "height": 768, "steps": 20, "seed": -1}
    max_side = 2048

    def validate(self, params: dict) -> dict:
        p = super().validate(params)
        if not str(p.get("prompt", "")).strip():
            raise UserError("Пустой промпт")
        for key in ("width", "height"):
            v = int(p[key])
            if not 256 <= v <= self.max_side:
                raise UserError(f"{key}: от 256 до {self.max_side} пикселей")
            p[key] = max(256, int(round(v / 64)) * 64)  # all models here want multiples of 16 or 64
        p["steps"] = int(p["steps"])
        if not 1 <= p["steps"] <= 150:
            raise UserError("steps: от 1 до 150")
        seed = int(p["seed"])
        p["seed"] = random.randint(0, 2**31 - 1) if seed < 0 else seed % 2**32
        if p.get("memory_mode", "auto") not in ("auto", "offload", "group", "sequential"):
            raise UserError("memory_mode: auto, offload, group или sequential")
        return p


class Img2MeshBackend(Backend):
    kind = "img2mesh"
    defaults = {"remove_bg": True, "bg_method": "auto", "mc_resolution": 256, "target_faces": 0}

    def validate(self, params: dict) -> dict:
        p = super().validate(params)
        res = int(p["mc_resolution"])
        if not 32 <= res <= 512:
            raise UserError("mc_resolution: от 32 до 512")
        p["mc_resolution"] = res
        p["target_faces"] = max(0, int(p.get("target_faces") or 0))
        if p["bg_method"] not in ("auto", "rembg", "simple", "alpha"):
            raise UserError("bg_method: auto, rembg, simple или alpha")
        return p


_CLASSES: list[type[Backend]] = []


def register(cls: type[Backend]) -> type[Backend]:
    _CLASSES.append(cls)
    return cls


class Registry:
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._items: dict[tuple[str, str], Backend] = {}
        for cls in _CLASSES:
            if settings.backend_enabled(cls.name):
                self._items[(cls.kind, cls.name)] = cls(settings)

    def get(self, kind: str, name: str) -> Backend:
        try:
            return self._items[(kind, name)]
        except KeyError:
            names = ", ".join(n for k, n in self._items if k == kind) or "нет"
            raise UserError(f"Нет бэкенда «{name}» для {kind}. Доступны: {names}") from None

    def of_kind(self, kind: str) -> list[Backend]:
        return [b for (k, _), b in self._items.items() if k == kind]

    def all(self) -> list[Backend]:
        return list(self._items.values())
