"""TripoSR (VAST-AI-Research + Stability AI, MIT): one picture -> textured-by-colour 3D mesh.

The code is TripoSR's own (a pinned clone on sys.path, see install.py) with two adaptations
that do not touch its files:

* ``torchmcubes`` is replaced by ``torchmcubes_shim`` (PyMCubes / scikit-image);
* transformers 5 renamed the ViT weights of its DINO image encoder, so the checkpoint keys
  are renamed before a strict ``load_state_dict`` (same rules as transformers' own
  ``conversion_mapping.py`` for ``ViTModel``).

VRAM is lowered through ``chunk_size`` (triplane queries per batch) and ``mc_resolution``; on
out-of-memory the job is retried with lighter settings, then on the CPU.
"""

from __future__ import annotations

import logging
import re
import sys

import numpy as np
from PIL import Image

from .. import cards, imaging, memory, meshing, weights
from ..errors import UserError, is_cuda_oom
from .base import Img2MeshBackend, register

log = logging.getLogger("dronegen.triposr")

VIT_PREFIX = "image_tokenizer.model."
# transformers 4 -> 5 names inside ViTModel; "attention.output.dense" must go before "output.dense"
VIT_RENAMES = [
    (re.compile(r"^encoder\.layer\.(\d+)\."), r"layers.\1."),
    (re.compile(r"attention\.attention\.query"), "attention.q_proj"),
    (re.compile(r"attention\.attention\.key"), "attention.k_proj"),
    (re.compile(r"attention\.attention\.value"), "attention.v_proj"),
    (re.compile(r"attention\.output\.dense"), "attention.o_proj"),
    (re.compile(r"intermediate\.dense"), "mlp.fc1"),
    (re.compile(r"(?<!attention\.)output\.dense"), "mlp.fc2"),
]
# TripoSR output axes: viewer at +X, picture right = +Y, up = +Z  ->  glTF: right +X, up +Y, viewer +Z
TO_GLTF = [1, 2, 0]
VRAM_GPU_GB = 2.5
RAM_NEED_GB = 4.0


def remap_vit_keys(state: dict, model_keys) -> dict:
    """Rename old-style ViT keys of the checkpoint when the model uses the new names."""
    model_keys = set(model_keys)
    out = {}
    for key, value in state.items():
        if key.startswith(VIT_PREFIX) and key not in model_keys:
            sub = key[len(VIT_PREFIX) :]
            for pattern, repl in VIT_RENAMES:
                sub = pattern.sub(repl, sub)
            key = VIT_PREFIX + sub
        out[key] = value
    return out


def to_gltf_axes(vertices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vertices[:, TO_GLTF])


@register
class TripoSR(Img2MeshBackend):
    name = "triposr"
    title = "TripoSR"
    needs_gpu = True
    heavy = True
    defaults = {**Img2MeshBackend.defaults, "mc_resolution": 256, "chunk_size": 8192, "foreground_ratio": 0.85}

    def __init__(self, settings):
        super().__init__(settings)
        self.model = None
        self.device = "cpu"

    def validate(self, params):
        p = super().validate(params)
        p["chunk_size"] = int(p["chunk_size"])
        if not 0 <= p["chunk_size"] <= 262144:
            raise UserError("chunk_size: от 0 до 262144 (0 — без разбиения)")
        p["foreground_ratio"] = float(p["foreground_ratio"])
        if not 0.5 <= p["foreground_ratio"] <= 1.0:
            raise UserError("foreground_ratio: от 0.5 до 1")
        return p

    def check(self):
        if memory.torch_or_none() is None:
            return False, "PyTorch не установлен (лёгкий профиль): используйте «Силуэт → объём»"
        try:
            import tsr  # noqa: F401
        except ImportError:
            return False, "код TripoSR не установлен — запустите python install.py"
        missing = weights.missing(self.settings, "triposr")
        return (False, weights.missing_message(missing)) if missing else (True, "")

    def load(self, ctx):
        import torch
        from omegaconf import OmegaConf

        from .. import torchmcubes_shim

        sys.modules["torchmcubes"] = torchmcubes_shim
        from tsr.system import TSR

        memory.require_ram(RAM_NEED_GB, "TripoSR", warn=ctx.warn)
        ctx.progress(0.02, "загрузка TripoSR")
        root = weights.folder(self.settings, "triposr")
        cfg = OmegaConf.load(root / "config.yaml")
        OmegaConf.resolve(cfg)
        model = TSR(cfg)
        state = torch.load(root / "model.ckpt", map_location="cpu", weights_only=True)
        state = remap_vit_keys(state, model.state_dict().keys())
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"веса TripoSR не подходят к модели: нет {missing[:4]}, лишние {unexpected[:4]}")
        model.eval()
        self.model = model
        self._place(ctx)

    def _place(self, ctx, force: str | None = None) -> None:
        """GPU when there is room, CPU otherwise (with a warning)."""
        want = force or ("cuda" if memory.cuda_ready() and memory.vram_free_gb() + self._on_gpu_gb() >= VRAM_GPU_GB else "cpu")
        if want == "cpu" and memory.cuda_ready() and force is None:
            ctx.warn("Мало свободной видеопамяти: TripoSR считает на процессоре — это в несколько раз дольше.")
        if want != self.device:
            self.model.to(want)
            self.device = want
            memory.free_gpu()

    def _on_gpu_gb(self) -> float:
        return 1.7 if self.device == "cuda" else 0.0

    def unload(self):
        self.model = None
        self.device = "cpu"
        memory.free_gpu()
        self.loaded = False

    def _infer(self, ctx, image: Image.Image, chunk: int, mc: int):
        import torch

        with torch.inference_mode():
            self.model.renderer.set_chunk_size(chunk)
            ctx.progress(0.2, "нейросеть TripoSR" + (" (CPU)" if self.device == "cpu" else ""))
            scene_codes = self.model([image], device=self.device)
            ctx.progress(0.55, f"извлечение поверхности ({mc}³)")
            return self.model.extract_mesh(scene_codes, True, resolution=mc)[0]

    def run(self, ctx, params, inputs):
        from tsr.utils import resize_foreground

        img = inputs["image"]
        ctx.progress(0.03, "удаление фона")
        if params["remove_bg"]:
            rgba, used = imaging.remove_background(img, params["bg_method"])
        elif imaging.has_alpha_matte(img):
            rgba, used = imaging.to_rgba(img), "alpha"
        else:
            rgba, used = None, "none"
        if rgba is not None:
            imaging.foreground_bbox(imaging.alpha_mask(rgba, 1))  # UserError when nothing is left
            rgba, count = cards.single_figure(rgba)
            if count > 1:
                ctx.warn(f"На картинке лист из {count} видов: для одноракурсной модели взята первая фигура (вид спереди)")
            rgba = resize_foreground(rgba, params["foreground_ratio"])
            arr = np.asarray(rgba).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5  # grey background, as trained
            image = Image.fromarray((arr * 255.0).astype(np.uint8))
        else:  # as TripoSR's --no-remove-bg: the picture must already be centred on grey
            image = img.convert("RGB")
        image.save(ctx.output("input", "png"))

        self._place(ctx)
        chunk, mc = params["chunk_size"], params["mc_resolution"]
        attempts = [(self.device, chunk, mc)]
        if self.device == "cuda":
            attempts += [("cuda", min(chunk or 8192, 2048), min(mc, 192)), ("cpu", 8192, min(mc, 192))]
        raw = None
        for i, (device, chunk_i, mc_i) in enumerate(attempts):
            try:
                if i:
                    self._place(ctx, force=device)
                    ctx.warn(f"Не хватило видеопамяти: повтор с chunk_size={chunk_i}, разрешением {mc_i}"
                             + (" на процессоре" if device == "cpu" else ""))
                raw = self._infer(ctx, image, chunk_i, mc_i)
                mc = mc_i
                break
            except Exception as e:  # noqa: BLE001
                if not is_cuda_oom(e) or i == len(attempts) - 1:
                    raise
                log.warning("TripoSR OOM (%s, chunk %s, mc %s)", device, chunk_i, mc_i)
                memory.free_gpu()

        ctx.progress(0.85, "сетка и цвета")
        colors = raw.visual.vertex_colors[:, :3].astype(np.float64) / 255.0 if raw.visual.kind == "vertex" else None
        mesh = meshing.make_mesh(to_gltf_axes(np.asarray(raw.vertices)), np.asarray(raw.faces), colors)
        mesh = meshing.clean(mesh)
        if meshing.signed_volume(mesh.vertices, mesh.faces) < 0:
            mesh.invert()
        mesh, note = meshing.decimate(mesh, params["target_faces"])
        if note:
            ctx.warn(note)
        meshing.normalize(mesh)
        ctx.progress(0.93, "экспорт")
        meshing.export(mesh, ctx.output("mesh", "glb"), ctx.output("mesh_obj", "obj"))
        return {**meshing.stats(mesh), "background": used, "device": self.device, "mc_resolution": mc}
