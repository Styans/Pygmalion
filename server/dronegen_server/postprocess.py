"""Mesh clean-up after any 3D model: debris, symmetry, smoothing, low-poly, thickness, colours.

Meshes are in glTF axes (+X right, +Y up, +Z towards the viewer). The voxel tools
(symmetry, smoothing) rebuild the surface from a filled voxel grid, which also removes the
«melted» marching-cubes lumps of neural models.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from . import imaging, meshing, preview, quality
from .cards import CAMERA

log = logging.getLogger("dronegen.postprocess")

STYLES = ("as_is", "smooth", "lowpoly")
# where a card may really have been drawn from: around the vertical (degrees from the front towards +X), from a little
# below to above the object; the pitches are symmetric, so that the mirror twin of every direction is tried too
CARD_YAWS = range(0, 360, 15)
CARD_PITCHES = (-30, -15, 0, 15, 30)
CARD_TURN_MARGIN = 0.05  # silhouette match a turned camera must win by: the card's own direction is the default
CARD_TIE = 0.02  # matches this close are equal: the one nearest the card's own direction is taken
SEEN_THROUGH_BLUR = 0.03  # a card guessed onto the far side is blurred by this share of the figure: its colours, not its details


def drop_debris(mesh, keep_ratio: float = 0.02):
    """Remove pieces whose surface is under ``keep_ratio`` of the largest one (floating specks)."""
    import trimesh

    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    largest = max(p.area for p in parts)
    keep = [p for p in parts if p.area >= keep_ratio * largest]
    return trimesh.util.concatenate(keep)


def _occupancy(mesh, resolution: int):
    """Filled boolean voxel grid of a watertight mesh, its pitch and origin."""
    pitch = float(mesh.extents.max()) / resolution
    vox = mesh.voxelized(pitch).fill()
    grid = np.asarray(vox.matrix, bool)
    origin = np.asarray(vox.transform)[:3, 3]  # centre of voxel [0, 0, 0]
    return grid, pitch, origin


def _mirrored(a: np.ndarray, c2: int) -> np.ndarray:
    """``a`` mirrored along axis 0 around index ``c2 / 2``; what falls outside is dropped."""
    n = a.shape[0]
    j = c2 - np.arange(n)
    ok = (j >= 0) & (j < n)
    out = np.zeros_like(a)
    out[ok] = a[j[ok]]
    return out


def symmetry_plane(grid: np.ndarray, search: float = 0.15) -> int:
    """Twice the x index of the left/right mirror plane of a voxel grid.

    It is the plane under which the front and top outlines best match their mirror images,
    searched around the centre of mass: a part on one side only (an arm, a sword) does not
    shift it, as it would shift the middle of the bounding box.
    """
    n = grid.shape[0]
    centre2 = int(round(2 * np.nonzero(grid)[0].mean()))
    outlines = (grid.any(axis=2), grid.any(axis=1))
    span = max(2, int(search * n))
    best, best_c2 = -1.0, centre2
    for c2 in range(centre2 - 2 * span, centre2 + 2 * span + 1):
        score = 0.0
        for o in outlines:
            m = _mirrored(o, c2)
            score += (o & m).sum() / max(1, (o | m).sum())
        if score > best:
            best, best_c2 = score, c2
    return best_c2


def voxel_rebuild(mesh, resolution: int = 200, mirror_x: bool = False, blur: float = 0.8):
    """Rebuild the surface from voxels: smooth and watertight.

    ``mirror_x``: also make it left/right symmetric - the union of the shape and its mirror
    image, so a part the model drew on one side only (a wing, a horn) appears on both.
    """
    from scipy import ndimage

    colors = mesh.visual.vertex_colors.copy() if mesh.visual.kind == "vertex" else None
    src = mesh
    grid, pitch, origin = _occupancy(mesh, resolution)
    if mirror_x:
        n = grid.shape[0]
        c2 = symmetry_plane(grid)
        lo, hi = max(0, n - 1 - c2), max(0, c2 - (n - 1))  # room for the mirror image
        grid = np.pad(grid, ((lo, hi), (0, 0), (0, 0)))
        grid |= _mirrored(grid, c2 + 2 * lo)
        origin = origin - np.array([lo * pitch, 0.0, 0.0])
    field = np.pad(grid.astype(np.float32), 2)
    if blur > 0:
        field = ndimage.gaussian_filter(field, blur)
    verts, faces = meshing.marching_cubes(field, 0.5)
    verts = (verts - 2) * pitch + origin
    out = meshing.make_mesh(verts, faces)
    out = drop_debris(out)
    meshing.smooth(out, 4)
    if colors is not None:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(src.vertices).query(out.vertices)
        out.visual.vertex_colors = colors[idx]
    return out


def lowpoly(mesh, faces: int = 4000):
    """Few flat faces, like a hand-made low-poly model (faceted shading in viewers)."""
    small, note = meshing.decimate(mesh, faces)
    small = small.copy()
    small.unmerge_vertices()
    return small, note


def set_depth(mesh, ratio: float):
    """Scale the thickness (Z) so that depth / width == ``ratio``."""
    current = mesh.extents[2] / max(mesh.extents[0], 1e-9)
    if current > 0:
        center = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2.0
        v = mesh.vertices.copy()
        v[:, 2] = center + (v[:, 2] - center) * (ratio / current)
        mesh.vertices = v
    return mesh


# ----------------------------------------------------------------------------------- colours
def _sample(card: Image.Image, mesh, direction) -> np.ndarray:
    """sRGB colour (0..1) of every vertex projected into a card, figure bbox to mesh bbox."""
    from scipy import ndimage

    rgba = np.asarray(imaging.to_rgba(card), np.float32) / 255.0
    alpha = rgba[..., 3] > 0.5
    t, b, l, r = imaging.foreground_bbox(alpha)
    # the rim of a cut-out is a blend with the old background (a light halo): do not trust it, take the
    # nearest colour from inside the figure instead; the same fill covers the background itself
    core = ndimage.binary_erosion(alpha, iterations=max(2, round(0.006 * max(alpha.shape))))
    _, (iy, ix) = ndimage.distance_transform_edt(~(core if core.any() else alpha), return_indices=True)
    rgb = rgba[..., :3][iy, ix]
    rr, uu, _ = preview._basis(direction)
    v = np.asarray(mesh.vertices)
    x, y = v @ rr, v @ uu
    cols = l + (x - x.min()) / max(np.ptp(x), 1e-9) * (r - l - 1)
    rows = t + (y.max() - y) / max(np.ptp(y), 1e-9) * (b - t - 1)
    return np.stack([ndimage.map_coordinates(rgb[..., c], [rows, cols], order=1, mode="nearest") for c in range(3)], 1)


def _soften(card: Image.Image, share: float = SEEN_THROUGH_BLUR) -> Image.Image:
    """The card with its small details washed out (eyes, a beak, spots, letters): only the colours of its large parts
    are left. Blurred inside the figure only, so the background and the light rim of the cut-out do not bleed in."""
    from scipy import ndimage

    rgba = np.asarray(imaging.to_rgba(card), np.float32)
    alpha = rgba[..., 3] > 127
    if not alpha.any():
        return card
    t, b, l, r = imaging.foreground_bbox(alpha)
    sigma = max(1.0, share * max(b - t, r - l))
    core = ndimage.binary_erosion(alpha, iterations=max(2, round(0.006 * max(alpha.shape))))
    inside = (core if core.any() else alpha).astype(np.float32)
    weight = np.maximum(ndimage.gaussian_filter(inside, sigma), 1e-6)
    rgb = np.stack([ndimage.gaussian_filter(rgba[..., c] * inside, sigma) / weight for c in range(3)], -1)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), rgba[..., 3]]).astype(np.uint8), "RGBA")


def _visible(mesh, direction, res: int | None = None) -> np.ndarray:
    """Which vertices a camera looking along ``-direction`` really sees (the rest hide behind other parts:
    a wing behind the body, the far side of a leg). A depth buffer made of the vertices themselves."""
    from scipy import ndimage

    rr, uu, _ = preview._basis(direction)
    v = np.asarray(mesh.vertices)
    if res is None:  # about one vertex per couple of pixels, so that the buffer has no holes
        res = int(np.clip(1.2 * np.sqrt(len(v)), 64, 768))
    x, y, depth = v @ rr, v @ uu, v @ np.asarray(direction, np.float64)
    scale = (res - 1) / max(np.ptp(x), np.ptp(y), 1e-9)
    px = np.clip(((x - x.min()) * scale).astype(int), 0, res - 1)
    py = np.clip(((y.max() - y) * scale).astype(int), 0, res - 1)
    zbuf = np.full((res, res), -np.inf)
    np.maximum.at(zbuf, (py, px), depth)
    zbuf = ndimage.maximum_filter(zbuf, size=3)  # vertices are sparse: close the gaps between them
    # a slope drops by about a pixel per pixel: two pixels of tolerance keep the sloped surface itself visible
    return depth >= zbuf[py, px] - 0.012 * max(np.ptp(depth), 1e-9) - 2.0 / scale


def _view_weight(mesh, normals, direction) -> np.ndarray:
    """How much a card can tell about each vertex: facing the camera, not hidden, not at a grazing angle
    (there one pixel is stretched over a long strip of the surface: the «stripes» on the sides)."""
    facing = np.clip((normals @ np.asarray(direction, np.float64) - 0.25) / 0.75, 0.0, 1.0) ** 2
    return facing * _visible(mesh, direction)


def _fill_unseen(mesh, colors: np.ndarray, known: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Vertices no card sees take the colour of the nearest seen ones, then are smoothed along the surface."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    if known.all() or not known.any():
        return colors
    v = np.asarray(mesh.vertices)
    dist, idx = cKDTree(v[known]).query(v[~known], k=min(10, int(known.sum())))
    w = 1.0 / (np.atleast_2d(dist) + 1e-6 * max(np.ptp(v), 1e-9))
    seen = colors[known]
    colors = colors.copy()
    colors[~known] = (seen[np.atleast_2d(idx)] * w[..., None]).sum(1) / w.sum(1)[:, None]
    e = np.asarray(mesh.edges_unique)
    n = len(v)
    adj = sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)[:, None]
    for _ in range(rounds):  # the filled area only: the seen colours stay as drawn
        smooth = adj @ colors / deg
        colors[~known] = 0.5 * colors[~known] + 0.5 * smooth[~known]
    return colors


def _direction(yaw: float, pitch: float) -> np.ndarray:
    """Camera direction turned ``yaw`` degrees from the front towards +X and raised ``pitch`` degrees."""
    a, p = np.radians(yaw), np.radians(pitch)
    return np.array([np.sin(a) * np.cos(p), np.sin(p), np.cos(a) * np.cos(p)])


def _points_mask(points: np.ndarray, direction, size: int) -> np.ndarray:
    """Silhouette of surface samples seen from ``direction``, fitted into a ``size`` square like ``quality._fit``."""
    from scipy import ndimage

    rr, uu, _ = preview._basis(direction)
    x, y = points @ rr, points @ uu
    k = (size - 1) / max(np.ptp(x), np.ptp(y), 1e-9)
    cols, rows = np.round((x - x.min()) * k).astype(int), np.round((y.max() - y) * k).astype(int)
    mask = np.zeros((rows.max() + 3, cols.max() + 3), bool)
    mask[rows + 1, cols + 1] = True
    return quality._fit(ndimage.binary_closing(mask)[1:-1, 1:-1], size)


def _card_mask(card: Image.Image, size: int) -> np.ndarray:
    figure = quality._crop(np.asarray(imaging.to_rgba(card))[..., 3] > 127)
    k = size / max(figure.shape)
    shape = (max(1, round(figure.shape[1] * k)), max(1, round(figure.shape[0] * k)))
    return quality._fit(np.asarray(Image.fromarray(figure).resize(shape, Image.NEAREST)), size)


def card_camera(mesh, name: str, card: Image.Image, points: np.ndarray | None = None, size: int = 128):
    """Direction the card was really drawn from: (direction, its silhouette match, the match from ``name``'s camera).

    Pictures are rarely exact views: a «front» is often a three-quarter view seen a little from above, a «right»
    may show the left side. Projected from the wrong side, the colours slide along the surface (the eyes on the
    bumper, a head on the tail). The card's own direction stays unless the mesh seen from another one matches
    its figure clearly better; among equal matches the nearest to its own direction wins, and on a tie the one
    seeing the front (figures are drawn facing us).
    """
    if points is None:
        points = np.asarray(mesh.sample(30000, seed=0))
    want = _card_mask(card, size)
    score = {(y, p): quality._iou(_points_mask(points, _direction(y, p), size), want)
             for y in CARD_YAWS for p in CARD_PITCHES}
    # a left/right symmetric object looks the same from (yaw, pitch) and from (180 - yaw, -pitch), e.g. from the
    # front and from the back: silhouettes cannot tell these twins apart, nearness to the card's own direction does
    score = {(y, p): max(s, score[(180 - y) % 360, -p]) for (y, p), s in score.items()}
    own = np.asarray(CAMERA[name], np.float64)
    at_own = score[round(np.degrees(np.arctan2(own[0], own[2]))) % 360, 0]
    best = max(score.values())
    if best < at_own + CARD_TURN_MARGIN:
        return own, at_own, at_own
    near = [g for g, s in score.items() if s >= best - CARD_TIE]
    g = max(near, key=lambda g: (round(float(_direction(*g) @ own), 6), _direction(*g)[2]))
    return _direction(*g), score[g], at_own


def colorize_from_views(mesh, views: dict[str, Image.Image]):
    """Vertex colours from the cards (``cards.CAMERA`` names).

    Every card is projected from where it was really drawn (``card_camera``), and every vertex takes its colour
    from the cards that see it well (facing, not hidden behind another part). A side nobody drew is guessed at half
    strength: the other side mirrored (right for most animals, vases, instruments), or for a front or a back the card
    seen through the object, blurred (``_soften``): the back of a head takes the colour of the head, not a second
    face with its eyes and beak; whatever no card sees is filled from its neighbours.
    """
    views = {k: v for k, v in views.items() if k in CAMERA and v is not None}
    if not views:
        return mesh
    points = np.asarray(mesh.sample(30000, seed=0))
    shots = []  # (card, camera direction, strength)
    for name, card in views.items():
        d, score, at_own = card_camera(mesh, name, card, points)
        if not np.allclose(d, CAMERA[name]):
            yaw, pitch = np.degrees(np.arctan2(d[0], d[2])), np.degrees(np.arcsin(np.clip(d[1], -1, 1)))
            log.info("card %s is projected from yaw %.0f, pitch %.0f: silhouette match %.2f against %.2f from its own side",
                     name, yaw, pitch, score, at_own)
        shots.append((card, d, 1.0))
    drawn = [d for _, d, _ in shots]
    for card, d, _ in list(shots):  # the guess for a side nobody drew
        mirror = abs(d[0]) >= 0.5 * np.hypot(d[0], d[2])  # a left or right side: the other side mirrored
        guess = d * (-1.0, 1.0, 1.0) if mirror else -d
        if all(g @ guess < np.cos(np.radians(45)) for g in drawn):
            flipped = card.transpose(Image.FLIP_LEFT_RIGHT)
            shots.append((flipped if mirror else _soften(flipped), guess, 0.5))
    normals = np.asarray(mesh.vertex_normals)
    total = np.zeros((len(mesh.vertices), 3))
    weight = np.zeros(len(mesh.vertices))
    for card, d, strength in shots:
        w = _view_weight(mesh, normals, d) * strength
        if not w.any():
            continue
        total += _sample(card, mesh, d) * w[:, None]
        weight += w
    known = weight > 0.02
    srgb = total / np.maximum(weight, 1e-9)[:, None]
    srgb = _fill_unseen(mesh, np.clip(srgb, 0, 1), known)
    lin = meshing.srgb_to_linear(np.clip(srgb, 0, 1))
    mesh.visual.vertex_colors = np.concatenate([np.round(lin * 255), np.full((len(lin), 1), 255)], axis=1).astype(np.uint8)
    return mesh


def apply(mesh, params: dict, views: dict[str, Image.Image] | None = None, warn=lambda m: None):
    """Everything the user asked for, in a fixed order; returns the new mesh.

    ``views``: cards to take the colours from (``cards.CAMERA`` names), None to keep them.
    """
    mesh = drop_debris(mesh)
    style = params.get("style", "as_is")
    symmetric = params.get("symmetry", False)
    if style == "smooth" or symmetric:
        try:
            mesh = voxel_rebuild(mesh, resolution=int(params.get("voxel_resolution", 200)), mirror_x=bool(symmetric))
        except Exception as e:  # noqa: BLE001 - a non-watertight input: keep the original
            warn(f"Сглаживание пропущено: {e}")
    if params.get("depth_ratio"):
        set_depth(mesh, float(params["depth_ratio"]))
    if views:
        colorize_from_views(mesh, views)
    if style == "lowpoly":
        mesh, note = lowpoly(mesh, int(params.get("lowpoly_faces", 4000)))
        if note:
            warn(note)
    elif params.get("target_faces"):
        mesh, note = meshing.decimate(mesh, int(params["target_faces"]))
        if note:
            warn(note)
    return mesh
