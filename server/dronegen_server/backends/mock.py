"""Test backends without neural networks: they check the whole pipeline on any machine."""

from __future__ import annotations

import math
import time

import numpy as np
from PIL import Image, ImageDraw

from .. import imaging, meshing
from .base import Img2MeshBackend, Txt2ImgBackend, register

PALETTE = [(220, 40, 40), (255, 170, 0), (30, 110, 220), (20, 160, 90), (150, 60, 200), (240, 200, 30)]
SHAPES = ("star", "ring", "crescent", "blob")


def draw_figure(width: int, height: int, seed: int) -> Image.Image:
    """A bold flat-colour figure on white, chosen by the seed (ring = a shape with holes)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    cx, cy = width / 2, height / 2
    r = 0.36 * min(width, height)
    a, b = (PALETTE[i] for i in rng.choice(len(PALETTE), 2, replace=False))
    shape = SHAPES[seed % len(SHAPES)]
    if shape == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            t = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
        d.polygon(pts, fill=a)
        d.ellipse([cx - r * 0.2, cy - r * 0.2, cx + r * 0.2, cy + r * 0.2], fill=b)
    elif shape == "ring":  # like a shanyrak: a rim with crossed arches and four openings
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=a)
        d.ellipse([cx - r * 0.72, cy - r * 0.72, cx + r * 0.72, cy + r * 0.72], fill="white")
        w = r * 0.14
        d.rectangle([cx - r * 0.75, cy - w, cx + r * 0.75, cy + w], fill=b)
        d.rectangle([cx - w, cy - r * 0.75, cx + w, cy + r * 0.75], fill=b)
    elif shape == "crescent":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=a)
        o = r * 0.45
        d.ellipse([cx - r + o, cy - r - o * 0.3, cx + r + o, cy + r - o * 0.3], fill="white")
        d.ellipse([cx - r * 0.55, cy - r * 0.1, cx - r * 0.25, cy + r * 0.2], fill=b)
    else:
        k = np.arange(1, 5)
        amp = rng.uniform(0.05, 0.18, len(k)) / k
        phase = rng.uniform(0, 2 * math.pi, len(k))
        t = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        rad = r * (0.8 + (amp[:, None] * np.cos(k[:, None] * t + phase[:, None])).sum(axis=0))
        d.polygon(list(zip(cx + rad * np.cos(t), cy + rad * np.sin(t))), fill=a)
        d.ellipse([cx - r * 0.25, cy - r * 0.25, cx + r * 0.25, cy + r * 0.25], fill=b)
    return img


@register
class MockTxt2Img(Txt2ImgBackend):
    name = "mock"
    title = "Тестовая картинка (без нейросети)"
    defaults = {**Txt2ImgBackend.defaults, "steps": 8}

    def run(self, ctx, params, inputs):
        steps = params["steps"]
        for i in range(steps):
            ctx.progress(0.9 * (i + 1) / steps, f"шаг {i + 1}/{steps}")
            time.sleep(0.03)
        img = draw_figure(params["width"], params["height"], params["seed"])
        meta = {k: params[k] for k in ("prompt", "negative", "width", "height", "steps", "seed")}
        meta.update({k: params[k] for k in ("subject", "style") if params.get(k)})
        imaging.save_png(img, ctx.output("image", "png"), {**meta, "backend": self.name})
        return {"seed": params["seed"], "prompt": params["prompt"], "width": img.width, "height": img.height}


@register
class MockImg2Mesh(Img2MeshBackend):
    name = "mock"
    title = "Тестовый меш (сфера цвета картинки)"
    defaults = {**Img2MeshBackend.defaults, "mc_resolution": 64}

    def run(self, ctx, params, inputs):
        import trimesh

        ctx.progress(0.3, "построение")
        rgb = np.asarray(inputs["image"].convert("RGB"), np.float64) / 255.0
        color = rgb.reshape(-1, 3).mean(axis=0)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh = meshing.make_mesh(mesh.vertices, mesh.faces, np.tile(color, (len(mesh.vertices), 1)))
        meshing.normalize(mesh)
        ctx.progress(0.8, "экспорт")
        meshing.export(mesh, ctx.output("mesh", "glb"), ctx.output("mesh_obj", "obj"))
        return meshing.stats(mesh)
