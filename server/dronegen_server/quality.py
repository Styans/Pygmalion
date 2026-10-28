"""Shape quality metrics for comparing 3D methods (docs/BENCHMARKS.md, «Точность 3D»).

* ``front_iou``  - how well the Front view matches the picture's figure (and its mirror image);
* ``depth_ratio`` - thickness / width (a dombra body is ~0.3-0.5, TripoSR made it 0.65);
* ``bumpiness``  - mean angle (deg) between a face normal and the average normal of its
  neighbourhood of radius 2 % of the size: the «melted» look, independent of face count;
* ``asymmetry``  - mean distance between the mesh and its left/right mirror, % of height.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import imaging, preview


def figure_mask(img: Image.Image) -> np.ndarray:
    rgba = imaging.to_rgba(img) if imaging.has_alpha_matte(img) else imaging.simple_matte(img)
    return imaging.alpha_mask(rgba)


def _crop(mask: np.ndarray) -> np.ndarray:
    t, b, l, r = imaging.foreground_bbox(mask)
    return mask[t:b, l:r]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def _fit(mask: np.ndarray, size: int) -> np.ndarray:
    """``mask`` scaled (keeping its proportions) so its longest side is ``size``, centred on a square."""
    out = np.zeros((size, size), bool)
    h, w = mask.shape
    out[(size - h) // 2 : (size - h) // 2 + h, (size - w) // 2 : (size - w) // 2 + w] = mask
    return out


def _view_masks(mesh, view, img: Image.Image, size: int):
    """The mesh seen from ``view`` and the picture's figure, each at its own proportions."""
    want = _crop(figure_mask(img))
    k = size / max(want.shape)
    shape = (max(1, round(want.shape[0] * k)), max(1, round(want.shape[1] * k)))
    want = np.asarray(Image.fromarray(want).resize(shape[::-1], Image.NEAREST))
    right, up, _ = preview._basis(view)
    v = np.asarray(mesh.vertices)
    w, h = np.ptp(v @ right), np.ptp(v @ up)
    k = size / max(w, h, 1e-9)
    got = preview.silhouette(mesh, view, (max(1, round(h * k)), max(1, round(w * k))))
    return _fit(got, size), _fit(want, size)


def view_iou(mesh, view, img: Image.Image, size: int = 160) -> float:
    """IoU of the mesh seen from ``view`` (name or camera direction) with the picture's figure."""
    got, want = _view_masks(mesh, view, img, size)
    return _iou(got, want)


def front_iou(mesh, img: Image.Image, size: int = 160) -> tuple[float, float]:
    """IoU of the mesh's Front view with the picture's figure, and with the mirrored figure."""
    got, want = _view_masks(mesh, "front", img, size)
    return _iou(got, want), _iou(got, want[:, ::-1])


def depth_ratio(mesh) -> float:
    ext = mesh.extents
    return float(ext[2] / max(ext[0], 1e-9))


def bumpiness(mesh, radius: float = 0.02, samples: int = 4000) -> float:
    from scipy.spatial import cKDTree

    centers = mesh.triangles_center
    normals = mesh.face_normals
    areas = mesh.area_faces
    size = float(mesh.extents.max())
    rng = np.random.default_rng(0)
    pick = rng.choice(len(centers), min(samples, len(centers)), replace=False)
    tree = cKDTree(centers)
    angles = []
    for i, nb in zip(pick, tree.query_ball_point(centers[pick], radius * size)):
        avg = (normals[nb] * areas[nb, None]).sum(axis=0)
        norm = np.linalg.norm(avg)
        if norm < 1e-12:
            continue
        angles.append(np.degrees(np.arccos(np.clip(normals[i] @ (avg / norm), -1.0, 1.0))))
    return float(np.mean(angles)) if angles else 0.0


def asymmetry(mesh) -> float:
    from scipy.spatial import cKDTree

    v = np.asarray(mesh.vertices)
    cx = (v[:, 0].min() + v[:, 0].max()) / 2.0
    mirrored = v.copy()
    mirrored[:, 0] = 2 * cx - mirrored[:, 0]
    d, _ = cKDTree(v).query(mirrored)
    return float(d.mean() / max(mesh.extents[1], 1e-9) * 100.0)


def report(mesh, img: Image.Image | None = None) -> dict:
    out = {
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "depth_ratio": round(depth_ratio(mesh), 3),
        "bumpiness_deg": round(bumpiness(mesh), 2),
        "asymmetry_pct": round(asymmetry(mesh), 2),
    }
    if img is not None:
        iou, mirrored = front_iou(mesh, img)
        out.update(front_iou=round(iou, 3), front_iou_mirrored=round(mirrored, 3))
    return out
