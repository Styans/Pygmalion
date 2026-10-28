"""Isosurfaces, mesh clean-up and export.

Every backend hands out meshes in one convention (glTF): +Y up, +X to the right of the
picture, +Z towards the viewer, centred, longest side 1. Blender's glTF importer turns
that into Z up with the picture facing -Y, i.e. the Front view shows the picture.
Vertex colours are written as linear COLOR_0, as glTF requires.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def _mc_pymcubes(volume: np.ndarray, level: float):
    import mcubes

    v, f = mcubes.marching_cubes(np.ascontiguousarray(volume, dtype=np.float64), float(level))
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def _mc_skimage(volume: np.ndarray, level: float):
    from skimage.measure import marching_cubes as sk_marching_cubes

    v, f, _n, _vals = sk_marching_cubes(np.asarray(volume, np.float32), level=float(level))
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def mc_backends() -> list[str]:
    names = []
    try:
        import mcubes  # noqa: F401

        names.append("pymcubes")
    except Exception:  # noqa: BLE001 - missing, or built against another numpy ABI
        pass
    names.append("skimage")
    return names


_MC = {"pymcubes": _mc_pymcubes, "skimage": _mc_skimage}


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume enclosed by the triangles; positive when they are wound outwards."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


@lru_cache(maxsize=None)
def _needs_flip(backend: str) -> bool:
    """Whether this library winds triangles inwards for 'inside = greater than level'."""
    n = 16
    g = np.mgrid[:n, :n, :n].astype(np.float64) - (n - 1) / 2
    ball = 5.0 - np.sqrt((g**2).sum(axis=0))
    v, f = _MC[backend](ball, 0.0)
    return signed_volume(v, f) < 0


def marching_cubes(volume: np.ndarray, level: float = 0.0, backend: str | None = None):
    """Isosurface where ``volume > level`` is inside.

    Returns vertices in index space (i, j, k), float64 (N, 3), and faces (M, 3) int64
    wound outwards. PyMCubes is used when installed, scikit-image otherwise.
    """
    backend = backend or mc_backends()[0]
    v, f = _MC[backend](volume, level)
    if len(f) and _needs_flip(backend):
        f = f[:, ::-1].copy()
    return v, f


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def make_mesh(verts: np.ndarray, faces: np.ndarray, colors_srgb: np.ndarray | None = None):
    """trimesh.Trimesh with vertex colours given as sRGB floats 0..1 (stored linear)."""
    import trimesh

    vc = None
    if colors_srgb is not None:
        lin = srgb_to_linear(np.clip(colors_srgb[:, :3], 0.0, 1.0))
        vc = np.concatenate([np.round(lin * 255.0), np.full((len(lin), 1), 255.0)], axis=1).astype(np.uint8)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vc, process=False)


def clean(mesh, min_component_faces: int = 50):
    """Drop tiny floating pieces and unused vertices (keeps the largest part at least)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        keep = [p for p in parts if len(p.faces) >= min_component_faces] or [max(parts, key=lambda p: len(p.faces))]
        import trimesh

        mesh = trimesh.util.concatenate(keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def smooth(mesh, iterations: int = 8):
    """Taubin smoothing: removes the voxel staircase without shrinking the shape."""
    import trimesh

    if iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=iterations)
    return mesh


def decimate(mesh, target_faces: int):
    """Fewer triangles; vertex colours are carried over from the nearest original vertex."""
    if not target_faces or len(mesh.faces) <= target_faces:
        return mesh, None
    try:
        small = mesh.simplify_quadric_decimation(face_count=int(target_faces))
    except Exception as e:  # noqa: BLE001 - optional dependency or degenerate input
        return mesh, f"упрощение сетки пропущено: {e}"
    if mesh.visual.kind == "vertex":
        from scipy.spatial import cKDTree

        _, idx = cKDTree(mesh.vertices).query(small.vertices)
        small.visual.vertex_colors = mesh.visual.vertex_colors[idx]
    return small, None


def normalize(mesh):
    """Centre the bounding box at the origin and scale the longest side to 1."""
    lo, hi = mesh.bounds
    size = float((hi - lo).max())
    if size <= 0:
        raise ValueError("пустая или вырожденная сетка")
    mesh.apply_translation(-(lo + hi) / 2.0)
    mesh.apply_scale(1.0 / size)
    return mesh


def stats(mesh) -> dict:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "volume": float(signed_volume(mesh.vertices, mesh.faces)),
        "extents": [round(float(x), 4) for x in mesh.extents],
    }


def export(mesh, glb_path: Path, obj_path: Path | None = None) -> None:
    """GLB (linear COLOR_0) and optionally OBJ (``v x y z r g b``, sRGB like other OBJ tools expect)."""
    mesh.export(glb_path, file_type="glb")
    if obj_path is None:
        return
    out = mesh
    if mesh.visual.kind == "vertex":
        out = mesh.copy()
        rgba = mesh.visual.vertex_colors.astype(np.float64) / 255.0
        rgba[:, :3] = linear_to_srgb(rgba[:, :3])
        out.visual.vertex_colors = np.round(rgba * 255.0).astype(np.uint8)
    out.export(obj_path, file_type="obj", include_color=True, include_normals=False, include_texture=False)
