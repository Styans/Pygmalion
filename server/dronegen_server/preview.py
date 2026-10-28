"""Software preview renders of a mesh (no GPU, no OpenGL): numpy + PIL, painter's algorithm.

Used by the Studio (thumbnails), by the assistant (it looks at the renders to judge a mesh)
and by the quality metrics. Meshes are in glTF axes: +X right, +Y up, +Z towards the viewer.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

# camera direction (from the object towards the camera) per view
VIEWS = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0),
    "three_quarter": (np.sin(np.radians(35)), np.sin(np.radians(15)), np.cos(np.radians(35))),
}
BACKGROUND = (38, 40, 48)
MAX_FACES = 60000


def _basis(view):
    """(right, up, towards the camera) for a view name or a camera direction vector."""
    f = np.asarray(VIEWS[view] if isinstance(view, str) else view, np.float64)
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, -1.0]) if abs(f[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return r, u, f


def _light(colors: np.ndarray, normals_cam: np.ndarray) -> np.ndarray:
    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    lambert = np.clip(normals_cam @ light, 0.0, 1.0)
    return np.clip(colors * (0.32 + 0.68 * lambert[:, None]), 0, 255)


def _face_colors(mesh) -> np.ndarray:
    """sRGB 0..255 per face (vertex colours are stored linear, as glTF requires)."""
    if getattr(mesh.visual, "kind", None) == "vertex":
        lin = mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, 1) ** (1 / 2.4) - 0.055)
        return srgb[mesh.faces].mean(axis=1) * 255.0
    return np.full((len(mesh.faces), 3), 200.0)


def _lighter(mesh):
    if len(mesh.faces) <= MAX_FACES:
        return mesh
    try:
        small = mesh.simplify_quadric_decimation(face_count=MAX_FACES)
    except Exception:  # noqa: BLE001 - preview only: fall back to a face subset
        keep = np.random.default_rng(0).choice(len(mesh.faces), MAX_FACES, replace=False)
        return mesh.submesh([keep], append=True)
    if getattr(mesh.visual, "kind", None) == "vertex":
        from scipy.spatial import cKDTree

        _, idx = cKDTree(mesh.vertices).query(small.vertices)
        small.visual.vertex_colors = mesh.visual.vertex_colors[idx]
    return small


def render(mesh, view="front", size: int = 512, background=BACKGROUND, margin: float = 0.06) -> Image.Image:
    """Orthographic, shaded render of ``mesh`` from ``view``, fitted into a ``size`` square."""
    mesh = _lighter(mesh)
    r, u, f = _basis(view)
    v = np.asarray(mesh.vertices, np.float64)
    center = (v.min(axis=0) + v.max(axis=0)) / 2.0
    v = v - center
    cam = np.stack([v @ r, v @ u, v @ f], axis=1)
    extent = max(np.ptp(cam[:, 0]), np.ptp(cam[:, 1]), 1e-9)
    scale = size * (1.0 - 2 * margin) / extent
    px = np.stack([size / 2 + cam[:, 0] * scale, size / 2 - cam[:, 1] * scale], axis=1)

    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    n_cam = np.stack([normals @ r, normals @ u, normals @ f], axis=1)
    visible = n_cam[:, 2] > -1e-6  # back faces are hidden by the front ones anyway
    depth = cam[faces, 2].mean(axis=1)
    order = np.argsort(depth[visible])  # far first
    idx = np.flatnonzero(visible)[order]
    colors = _light(_face_colors(mesh)[idx], n_cam[idx]).astype(np.uint8)

    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)
    tri = px[faces[idx]]
    for poly, color in zip(tri, colors):
        c = tuple(int(x) for x in color)
        draw.polygon([tuple(p) for p in poly], fill=c, outline=c)
    return img


def sheet(mesh, views=("front", "three_quarter", "right"), size: int = 384) -> Image.Image:
    """Several views side by side (for the assistant's checks and the Studio)."""
    tiles = [render(mesh, v, size) for v in views]
    out = Image.new("RGB", (size * len(tiles), size), BACKGROUND)
    for i, t in enumerate(tiles):
        out.paste(t, (i * size, 0))
    return out


def silhouette(mesh, view, shape: tuple[int, int]) -> np.ndarray:
    """Mask of the mesh seen from ``view`` (name or camera direction), spanning its bounding box (row 0 = top)."""
    h, w = shape
    r, u, f = _basis(view)
    v = np.asarray(mesh.vertices, np.float64)
    x, y = v @ r, v @ u
    cols = (x - x.min()) / max(np.ptp(x), 1e-9) * w
    rows = (y.max() - y) / max(np.ptp(y), 1e-9) * h
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for face in np.asarray(mesh.faces):
        draw.polygon([(cols[i], rows[i]) for i in face], fill=255)
    return np.asarray(img) > 0


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
