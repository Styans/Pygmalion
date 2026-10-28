"""Silhouette -> inflated 2.5D volume. No neural network for the 3D part, no GPU.

1. Mask of the figure (alpha / rembg / plain-background cut, see ``imaging``).
2. Signed distance ``S`` to the outline (inside > 0) and the local half-width ``W``: the
   radius of the largest inscribed circle that covers the pixel (morphological openings).
3. Height ``h = sqrt(W² − (W − S)²)``: every limb gets a round cross-section, and the height
   is softly capped at ``depth × size / 2``, so wide parts become a pillow, not a ball.
4. Field ``F(x, y, z) = h(x, y) − |z|`` (outside the outline: ``S − |z|``), marching cubes at
   0 and Taubin smoothing. The result is closed (watertight) and symmetric front/back.
5. Vertex colours: the picture projected from the front.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import cards, imaging, meshing
from ..errors import UserError
from .base import Img2MeshBackend, register

PAD = 3  # voxels of empty space around the figure, so the surface closes
PROFILES = ("round", "slab")


@register
class Silhouette(Img2MeshBackend):
    name = "silhouette"
    title = "Силуэт → объём (без нейросети, CPU)"
    defaults = {**Img2MeshBackend.defaults, "mc_resolution": 192, "depth": 0.3, "smooth": 8, "profile": "round"}

    def validate(self, params):
        p = super().validate(params)
        p["depth"] = float(p["depth"])
        if not 0.01 <= p["depth"] <= 1.0:
            raise UserError("depth (толщина): от 0.01 до 1")
        p["smooth"] = int(p["smooth"])
        if not 0 <= p["smooth"] <= 50:
            raise UserError("smooth: от 0 до 50")
        if p["profile"] not in PROFILES:
            raise UserError("profile: round (округлый объём) или slab (плита: точный вырез по контуру)")
        return p

    def run(self, ctx, params, inputs):
        img = inputs["image"]
        ctx.progress(0.02, "удаление фона")
        if params["remove_bg"]:
            rgba, used = imaging.remove_background(img, params["bg_method"])
        elif imaging.has_alpha_matte(img):
            rgba, used = imaging.to_rgba(img), "alpha"
        else:
            rgba, used = imaging.simple_matte(img), "simple"
        rgba, count = cards.single_figure(rgba)
        if count > 1:
            ctx.warn(f"На картинке лист из {count} видов: для одноракурсной модели взята первая фигура (вид спереди)")
        rgba.save(ctx.output("cutout", "png"))
        mesh = inflate(
            rgba,
            resolution=params["mc_resolution"],
            depth=params["depth"],
            smooth=params["smooth"],
            profile=params["profile"],
            progress=lambda f, stage: ctx.progress(0.15 + 0.75 * f, stage),
        )
        mesh, note = meshing.decimate(mesh, params["target_faces"])
        if note:
            ctx.warn(note)
        meshing.normalize(mesh)
        ctx.progress(0.93, "экспорт")
        meshing.export(mesh, ctx.output("mesh", "glb"), ctx.output("mesh_obj", "obj"))
        return {**meshing.stats(mesh), "background": used}


def _resize_float(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float field (with anti-aliasing when shrinking)."""
    im = Image.fromarray(np.ascontiguousarray(a, np.float32), mode="F")
    return np.asarray(im.resize((shape[1], shape[0]), Image.BILINEAR), np.float32)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop specks and fill pinholes (relative to the figure area); real holes stay."""
    from scipy import ndimage

    area = int(mask.sum())
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= max(4, 0.002 * area)) + 1
        if keep.size == 0:
            keep = np.array([int(np.argmax(sizes)) + 1])
        mask = np.isin(lab, keep)
    holes, m = ndimage.label(~mask)
    if m > 1:
        border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
        sizes = ndimage.sum(~mask, holes, index=np.arange(1, m + 1))
        small = [i + 1 for i, s in enumerate(sizes) if s < 0.0005 * area and (i + 1) not in border]
        if small:
            mask = mask | np.isin(holes, small)
    return mask


def _local_half_width(S: np.ndarray) -> np.ndarray:
    """Radius of the largest inscribed circle covering each inside pixel (0 outside)."""
    from scipy import ndimage

    inside = S > 0
    smax = float(S.max())
    W = np.where(inside, np.maximum(S, 0.5), 0.0).astype(np.float32)
    if smax <= 1.0:
        return W
    radii = np.unique(np.round(np.concatenate([np.arange(1.0, min(smax, 8.0), 1.0), np.geomspace(1.0, smax, 28)]), 3))
    for r in radii:
        core = S > r
        if not core.any():
            break
        covered = ndimage.distance_transform_edt(~core) <= r  # union of inscribed disks of radius r
        W = np.where(covered & inside, np.maximum(W, r), W)
    return W


def _soft_min(a: np.ndarray, b: float, k: float) -> np.ndarray:
    """min(a, b) with a rounded corner of size ~k (no crease where a limb reaches the cap)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def height_field(mask: np.ndarray, resolution: int, depth: float, profile: str = "round"):
    """(H, colour-grid shape helpers) for a cleaned boolean mask cropped to the figure.

    Returns S (signed distance, voxels), H (height inside, S outside) on a grid whose
    longest side is ``resolution`` voxels including the padding.
    """
    from scipy import ndimage

    content = resolution - 2 * PAD
    h_px, w_px = mask.shape
    scale = content / max(h_px, w_px)  # voxels per source pixel
    margin = int(np.ceil(PAD / scale)) + 1
    padded = np.pad(mask, margin)
    S_hi = ndimage.distance_transform_edt(padded) - ndimage.distance_transform_edt(~padded)
    gh = max(2 * PAD + 2, int(round(padded.shape[0] * scale)))
    gw = max(2 * PAD + 2, int(round(padded.shape[1] * scale)))
    S = _resize_float(S_hi, (gh, gw)) * scale
    inside = S > 0
    if not inside.any():
        raise UserError("Фигура слишком маленькая для выбранного разрешения — увеличьте разрешение сетки")
    W = _local_half_width(S)
    W = np.maximum(ndimage.gaussian_filter(W, 1.0), np.maximum(S, 0.0))
    Sc = np.clip(S, 0.0, None)
    cap = max(1.0, depth * content / 2.0)
    if profile == "slab":  # a cut-out plate: steep walls up to the full thickness, a small chamfer
        h = _soft_min(Sc * 6.0, cap, 0.04 * cap)
    else:  # every limb gets a round cross-section, wide parts become a pillow
        tube = np.sqrt(np.clip(Sc * (2.0 * W - Sc), 0.0, None))
        h = _soft_min(tube, cap, 0.15 * cap)
    H = np.where(inside, h, S).astype(np.float32)
    return S, H, margin, scale


def inflate(rgba: Image.Image, resolution: int = 192, depth: float = 0.3, smooth: int = 8, progress=None,
            profile: str = "round"):
    """Closed, vertex-coloured trimesh in glTF axes (+Y up, +Z towards the viewer)."""
    from scipy import ndimage

    progress = progress or (lambda f, s: None)
    rgba = imaging.to_rgba(rgba)
    # work at no more than 4x the grid resolution: plenty for a smooth outline
    limit = 4 * resolution
    if max(rgba.size) > limit:
        k = limit / max(rgba.size)
        rgba = rgba.resize((max(1, round(rgba.width * k)), max(1, round(rgba.height * k))), Image.LANCZOS)
    mask = imaging.alpha_mask(rgba)
    top, bottom, left, right = imaging.foreground_bbox(mask)
    mask = _clean_mask(mask[top:bottom, left:right])
    rgb = np.asarray(rgba, np.float32)[top:bottom, left:right, :3] / 255.0

    progress(0.1, "контур и толщина")
    S, H, margin, scale = height_field(mask, resolution, depth, profile)

    # colours on the same grid; the background takes the nearest figure colour (no white rims)
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    rgb = np.pad(rgb[iy, ix], ((margin, margin), (margin, margin), (0, 0)), mode="edge")
    grid_rgb = np.stack([_resize_float(rgb[..., c], H.shape) for c in range(3)], axis=-1)

    progress(0.35, "объём")
    zmax = int(np.ceil(float(H.max()))) + 2
    z = np.arange(-zmax, zmax + 1, dtype=np.float32)
    field = H[:, :, None] - np.abs(z)[None, None, :]

    progress(0.5, "поверхность (marching cubes)")
    verts, faces = meshing.marching_cubes(field, 0.0)
    del field
    if len(faces) == 0:
        raise UserError("Не удалось построить поверхность: фигура слишком тонкая для выбранного разрешения")
    i, j, k = verts[:, 0], verts[:, 1], verts[:, 2]
    colors = np.stack([ndimage.map_coordinates(grid_rgb[..., c], [i, j], order=1, mode="nearest") for c in range(3)], 1)
    # index space (row, col, layer) -> x right, y up, z towards the viewer: a proper rotation,
    # so the outward winding from marching_cubes is kept
    xyz = np.stack([j, -i, k - zmax], axis=1)

    progress(0.75, "сглаживание")
    mesh = meshing.make_mesh(xyz, faces, colors)
    mesh = meshing.clean(mesh)
    meshing.smooth(mesh, smooth)
    progress(1.0, "готово")
    return mesh
