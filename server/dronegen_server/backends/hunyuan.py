"""Hunyuan3D-2 (Tencent): picture(s) -> a clean, precise 3D shape.

* ``hunyuan``    - Hunyuan3D-2mini-turbo, one picture;
* ``hunyuan_mv`` - Hunyuan3D-2mv-turbo, several views («карточки») of one object: front,
  left, back, right (``cards.CAMERA``), or a turnaround sheet read into front, side and back
  (``cards.read_sheet``: by what the figures show, the image model repeats and mixes views).
  With a side view the shape error drops ~3.4x against one picture (docs/BENCHMARKS.md).

Shape only: Hunyuan's own texturing needs custom CUDA extensions. Vertex colours are
projected from the pictures instead (``postprocess.colorize_from_views``).

The code is Tencent's ``hy3dgen`` package (a pinned commit on sys.path, see install.py) with
adaptations that do not touch its files:

* ``pymeshlab`` (GPL; only their optional post-processors import it) is replaced by a stub;
* transformers 5 renamed the DINOv2 weights of the image encoder: the checkpoint keys are
  renamed before loading (``hfcompat.remap_dinov2``);
* weights are read from ``<models>/hy3dgen`` (``HY3DGEN_MODELS``), never downloaded here.

The output is already in glTF axes (+Y up, the picture seen from +Z, picture right = +X).

Licence: Tencent Hunyuan 3D 2.0 Community License (README, «Лицензии»): not for use in the EU,
the UK and South Korea; outputs may not be used to improve other AI models.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
from PIL import Image

from .. import cards, hfcompat, imaging, memory, meshing, postprocess, preview, quality, weights
from ..errors import UserError, is_cuda_oom
from .base import Img2MeshBackend, register

log = logging.getLogger("dronegen.hunyuan")

RAM_NEED_GB = 6.0
VRAM_NEED_GB = {"hunyuan": 5.3, "hunyuan_mv": 6.1}  # peak, measured on an RTX 3050 8 GB
TRIAL_RESOLUTION = 128  # quick surface to tell which side a side card shows
SIDE_MARGIN = 0.05  # below this difference of the two tries the choice is reported as unsure
SIDES = ("auto", "left", "right")


def _stub_pymeshlab() -> None:
    """hy3dgen imports pymeshlab at the top of its post-processors, which are not used here."""
    try:
        import pymeshlab  # noqa: F401

        return
    except ImportError:
        pass
    stub = types.ModuleType("pymeshlab")

    def missing(name):
        if name.startswith("__"):
            raise AttributeError(name)

        def init(self, *a, **k):
            raise RuntimeError("pymeshlab не установлен: постобработка Hunyuan3D не используется")

        return type(name, (), {"__init__": init})

    stub.__getattr__ = missing
    sys.modules["pymeshlab"] = stub


_patched = False


def import_pipeline():
    """``Hunyuan3DDiTFlowMatchingPipeline`` with the compatibility patches applied (once)."""
    global _patched
    _stub_pymeshlab()
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.models import conditioner

    if not _patched:
        original = conditioner.SingleImageEncoder.load_state_dict

        def load_state_dict(self, state_dict, strict=True, assign=False):
            state_dict = hfcompat.remap_dinov2(state_dict, self.state_dict().keys())
            return original(self, state_dict, strict=strict, assign=assign)

        conditioner.SingleImageEncoder.load_state_dict = load_state_dict
        _patched = True
    return Hunyuan3DDiTFlowMatchingPipeline


def cutout(img: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """The figure on a transparent background, and how the background was removed."""
    if params["remove_bg"]:
        rgba, used = imaging.remove_background(img, params["bg_method"])
    elif imaging.has_alpha_matte(img):
        rgba, used = imaging.to_rgba(img), "alpha"
    else:
        raise UserError("Hunyuan3D нужен вырезанный объект: включите удаление фона")
    imaging.foreground_bbox(imaging.alpha_mask(rgba, 16))  # UserError when nothing is left
    return rgba, used


@register
class Hunyuan(Img2MeshBackend):
    name = "hunyuan"
    title = "Hunyuan3D-2mini (точная форма по картинке)"
    needs_gpu = True
    heavy = True
    multiview = False
    weights_group = "hunyuan_mini"
    repo, subfolder = "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"
    defaults = {
        **Img2MeshBackend.defaults,
        "mc_resolution": 384,  # octree resolution of the surface
        "steps": 5,
        "guidance": 5.0,
        "seed": 1,
        "num_chunks": 20000,
        "style": "as_is",  # as_is | smooth | lowpoly (postprocess)
        "symmetry": False,  # mirror-average left and right
        "depth_ratio": 0.0,  # thickness / width; 0: as the model made it
        "colors": True,  # project the pictures onto the surface
    }

    def __init__(self, settings):
        super().__init__(settings)
        self.pipe = None

    # -- checks -------------------------------------------------------------------------
    def validate(self, params):
        p = super().validate(params)
        p["steps"] = int(p["steps"])
        if not 1 <= p["steps"] <= 50:
            raise UserError("steps: от 1 до 50")
        p["guidance"] = float(p["guidance"])
        if not 0 <= p["guidance"] <= 20:
            raise UserError("guidance: от 0 до 20")
        p["seed"] = int(p["seed"]) % 2**32
        p["num_chunks"] = int(p["num_chunks"])
        if not 1000 <= p["num_chunks"] <= 500000:
            raise UserError("num_chunks: от 1000 до 500000")
        if p["style"] not in postprocess.STYLES:
            raise UserError("style: " + ", ".join(postprocess.STYLES))
        p["symmetry"] = bool(p["symmetry"])
        p["depth_ratio"] = float(p["depth_ratio"] or 0.0)
        if p["depth_ratio"] and not 0.02 <= p["depth_ratio"] <= 3.0:
            raise UserError("depth_ratio (толщина к ширине): от 0.02 до 3, или 0 — как получилось")
        p["colors"] = bool(p["colors"])
        return p

    def check(self):
        if memory.torch_or_none() is None:
            return False, "PyTorch не установлен (лёгкий профиль): используйте «Силуэт → объём»"
        if not memory.cuda_ready():
            return False, "нужна видеокарта NVIDIA (CUDA)"
        try:
            import hy3dgen  # noqa: F401
        except ImportError:
            return False, "код Hunyuan3D-2 не установлен — запустите python install.py"
        missing = weights.missing(self.settings, self.weights_group)
        return (False, weights.missing_message(missing)) if missing else (True, "")

    # -- lifecycle ----------------------------------------------------------------------
    def load(self, ctx):
        import torch

        memory.require_ram(RAM_NEED_GB, self.title, warn=ctx.warn)
        need = VRAM_NEED_GB[self.name]
        if memory.vram_free_gb() < need:
            ctx.warn(memory.vram_hint(need))
        ctx.progress(0.02, "загрузка Hunyuan3D")
        pipeline = import_pipeline()
        pipe = pipeline.from_pretrained(
            self.repo, subfolder=self.subfolder, use_safetensors=True, variant="fp16", device="cuda", dtype=torch.float16
        )
        ctx.progress(0.08, "загрузка быстрого декодера (FlashVDM)")
        pipe.enable_flashvdm(mc_algo="mc")
        self.pipe = pipe
        memory.free_gpu()
        self.loaded = True

    def unload(self):
        self.pipe = None
        memory.free_gpu()
        self.loaded = False

    # -- generation ---------------------------------------------------------------------
    def views(self, ctx, params, inputs) -> tuple[dict[str, Image.Image], str]:
        """Background-free cards by ``cards.CAMERA`` name (plus "side": not known which)."""
        rgba, used = cutout(inputs["image"], params)
        rgba, count = cards.single_figure(rgba)
        if count > 1:
            ctx.warn(f"На картинке лист из {count} видов: для одноракурсной модели взята первая фигура (вид спереди)")
        return {"front": rgba}, used

    def diffuse(self, ctx, params, views: dict[str, Image.Image], span=(0.12, 0.5)):
        """Latent shape for the cards (the slow, neural part)."""
        import torch

        steps, (lo, hi) = params["steps"], span

        def on_step(i, _t, _out):
            ctx.progress(lo + (hi - lo) * (i + 1) / steps, f"нейросеть: шаг {i + 1} из {steps}")

        ctx.progress(lo, "нейросеть Hunyuan3D")
        return self.pipe(
            image=dict(views) if self.multiview else views["front"],
            num_inference_steps=steps,
            guidance_scale=params["guidance"],
            generator=torch.Generator("cuda").manual_seed(params["seed"]),
            output_type="latent",
            enable_pbar=False,
            callback=on_step,
            callback_steps=1,
        )

    def decode(self, ctx, latents, resolution: int, chunks: int):
        """Surface of a latent shape as a trimesh in glTF axes; lighter retry on out-of-memory."""
        import torch

        attempts = [(resolution, chunks), (min(resolution, 256), min(chunks, 8000))]
        for i, (res, ch) in enumerate(attempts):
            try:
                if i:
                    ctx.warn(f"Не хватило видеопамяти: поверхность с разрешением {res}")
                with torch.inference_mode():
                    out = self.pipe._export(
                        latents, output_type="trimesh", box_v=1.01, mc_level=0.0, num_chunks=ch,
                        octree_resolution=res, mc_algo=None, enable_pbar=False,
                    )
                break
            except Exception as e:  # noqa: BLE001
                if not is_cuda_oom(e) or i == len(attempts) - 1:
                    raise
                log.warning("Hunyuan3D decode OOM at resolution %s", res)
                memory.free_gpu()
        mesh = out[0]
        if mesh is None or len(mesh.faces) == 0:
            raise UserError("Hunyuan3D не нашёл поверхность: попробуйте другую картинку или seed")
        mesh = meshing.make_mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64))
        if meshing.signed_volume(mesh.vertices, mesh.faces) < 0:
            mesh.invert()
        return mesh

    def place_side(self, ctx, params, views: dict[str, Image.Image]):
        """Put a side card of unknown side where it fits: (views, latents or None, note).

        Both ways are tried. Placed right, the card agrees with the others and the shape
        matches every card; placed wrong, the model either half-ignores it or builds the
        object mirrored in depth, which shows as a worse match (on the test object: 0.94
        against 0.78 with two cards, 0.91 against 0.46 with three). Silhouettes alone cannot
        tell a shape from its depth mirror: the shading and colours of the pictures decide.
        """
        side = views.pop("side", None)
        if side is None:
            return views, None, ""
        if params["side"] != "auto":
            return {**views, params["side"]: side}, None, ""
        scores = {}
        for i, which in enumerate(("left", "right")):
            trial = {**views, which: side}
            latents = self.diffuse(ctx, params, trial, span=(0.12 + 0.2 * i, 0.3 + 0.2 * i))
            rough = self.decode(ctx, latents, TRIAL_RESOLUTION, params["num_chunks"])
            fit = np.mean([quality.view_iou(rough, cards.CAMERA[name], card) for name, card in trial.items()])
            scores[which] = (float(fit), trial, latents)
        best = max(scores, key=lambda k: scores[k][0])
        other = "right" if best == "left" else "left"
        word = {"left": "слева", "right": "справа"}
        note = (f"карточка «сбоку» — вид {word[best]}: совпадение с карточками {scores[best][0]:.2f} "
                f"против {scores[other][0]:.2f}")
        log.info(note)
        if scores[best][0] - scores[other][0] < SIDE_MARGIN:
            ctx.warn(f"Не удалось уверенно понять, какой бок на карточке «сбоку» ({note}). "
                     "Если фигура вышла вывернутой, укажите сторону вручную (side: left или right)")
        return scores[best][1], scores[best][2], note

    def run(self, ctx, params, inputs):
        ctx.progress(0.03, "подготовка карточек")
        views, used = self.views(ctx, params, inputs)
        views, latents, note = self.place_side(ctx, params, views)
        for name, card in views.items():
            card.save(ctx.output(f"card_{name}", "png"))
        if latents is None:
            latents = self.diffuse(ctx, params, views)
        ctx.progress(0.55, f"поверхность ({params['mc_resolution']})")
        mesh = self.decode(ctx, latents, params["mc_resolution"], params["num_chunks"])

        ctx.progress(0.72, "чистка сетки" + (" и цвета" if params["colors"] else ""))
        mesh = postprocess.apply(mesh, params, views if params["colors"] else None, warn=ctx.warn)
        meshing.normalize(mesh)

        ctx.progress(0.9, "превью и экспорт")
        preview.sheet(mesh).save(ctx.output("preview", "png"))
        meshing.export(mesh, ctx.output("mesh", "glb"), ctx.output("mesh_obj", "obj"))
        result = {**meshing.stats(mesh), "background": used, "views": list(views)}
        result["quality"] = quality.report(mesh, views["front"])
        if note:
            result["side_note"] = note
        return result


@register
class HunyuanMV(Hunyuan):
    name = "hunyuan_mv"
    title = "Hunyuan3D-2mv (форма по карточкам видов)"
    multiview = True
    weights_group = "hunyuan_mv"
    repo, subfolder = "tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo"
    defaults = {**Hunyuan.defaults, "side": "auto"}

    def validate(self, params):
        p = super().validate(params)
        if p["side"] not in SIDES:
            raise UserError("side (какой бок показывает карточка «сбоку»): auto, left или right")
        return p

    def views(self, ctx, params, inputs):
        """Cards from ``inputs["views"]`` ({front, left, back, right, side}) or a turnaround sheet."""
        given = dict(inputs.get("views") or {})
        methods = set()
        from_sheet = not given
        if from_sheet:
            if "image" not in inputs:
                raise UserError("Нужны карточки видов или лист видов (спереди | слева | сзади | справа)")
            ctx.progress(0.04, "нарезка листа видов на карточки")
            sheet, used = cutout(inputs["image"], params)
            read = cards.read_sheet(sheet)
            if read.notes:
                ctx.warn(f"Лист видов: {read.report()}")
            given = read.views
            methods.add(used)
        unknown = set(given) - set(cards.CAMERA) - {"side"}
        if unknown:
            raise UserError(f"Неизвестные виды: {', '.join(sorted(unknown))}. Можно: front, left, back, right, side")
        if "front" not in given:
            raise UserError("Нужна карточка «спереди» (front)")
        if "side" in given and ("left" in given or "right" in given):
            raise UserError("Карточка «side» нужна, только когда неизвестно, левый это бок или правый")
        if len(given) < 2 and not from_sheet:
            raise UserError("Для Hunyuan3D-2mv нужно хотя бы 2 карточки: спереди и сбоку или сзади")
        out = {}
        for name, img in given.items():
            out[name], used = cutout(img, params)
            methods.add(used)
        return out, "+".join(sorted(methods))
