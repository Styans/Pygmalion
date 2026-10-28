"""Building blocks for exact (parametric) models: rings joined into closed solids, solids of
revolution, tubes along curves, boxes and cylinders, each part with its own colour.

Meshes are in glTF axes (+Y up, +Z towards the viewer), real sizes in metres. Every part is a
closed, outward-facing surface; parts are simply put together (no boolean union), which is what
point sampling for drone formations needs.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .. import meshing

Color = tuple[float, float, float]  # sRGB 0..1


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def solid(mesh: trimesh.Trimesh, color: Color) -> trimesh.Trimesh:
    """The same mesh with one vertex colour (sRGB in, stored linear as glTF wants)."""
    out = meshing.make_mesh(mesh.vertices, mesh.faces, np.tile(np.asarray(color, np.float64), (len(mesh.vertices), 1)))
    return out


def recolor(mesh: trimesh.Trimesh, picker) -> trimesh.Trimesh:
    """Vertex colours chosen per vertex: ``picker(vertices) -> (n, 3) sRGB``."""
    colors = np.clip(np.asarray(picker(mesh.vertices), np.float64), 0.0, 1.0)
    return meshing.make_mesh(mesh.vertices, mesh.faces, colors)


def join(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate([p for p in parts if p is not None and len(p.faces)])


def loft(rings: list[np.ndarray], closed: bool = False, caps: bool = True) -> trimesh.Trimesh:
    """Rings (all with the same number of points) joined one to the next, outward-facing.

    ``closed``: the last ring joins the first (a torus); otherwise the two ends are capped.
    """
    rings = [np.asarray(r, np.float64) for r in rings]
    m, n = len(rings), len(rings[0])
    if any(len(r) != n for r in rings):
        raise ValueError("все кольца должны иметь одинаковое число точек")
    verts = np.vstack(rings)
    quads = []
    for i in range(m if closed else m - 1):
        i2 = (i + 1) % m
        for j in range(n):
            j2 = (j + 1) % n
            quads.append((i * n + j, i * n + j2, i2 * n + j2, i2 * n + j))
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    if not closed:  # the first quad's normal must point away from the centroid of its ring
        a, b, c = verts[faces[0, 0]], verts[faces[0, 1]], verts[faces[0, 2]]
        if np.cross(b - a, c - a) @ (a - rings[0].mean(axis=0)) < 0:
            faces = faces[:, ::-1]
        if caps:
            c0, c1 = rings[0].mean(axis=0), rings[-1].mean(axis=0)
            axis = c1 - c0
            i0, i1 = len(verts), len(verts) + 1
            verts = np.vstack([verts, c0[None], c1[None]])
            cap0 = np.array([[i0, j, (j + 1) % n] for j in range(n)])
            cap1 = np.array([[i1, (m - 1) * n + j, (m - 1) * n + (j + 1) % n] for j in range(n)])
            for cap, outward in ((cap0, -axis), (cap1, axis)):
                tri = verts[cap]
                if np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0) @ outward < 0:
                    cap[:] = cap[:, ::-1]
            faces = np.vstack([faces, cap0, cap1])
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if closed and meshing.signed_volume(mesh.vertices, mesh.faces) < 0:  # a torus has no single outward test
        mesh.invert()
    return mesh


def circle(center, radius: float, normal, sections: int = 24, phase: float = 0.0) -> np.ndarray:
    normal = unit(normal)
    ref = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(normal, ref))
    v = np.cross(normal, u)
    t = np.linspace(0, 2 * np.pi, sections, endpoint=False) + phase
    return np.asarray(center) + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)


def lathe(profile: list[tuple[float, float]], sections: int = 48) -> trimesh.Trimesh:
    """Solid of revolution around the Y axis. ``profile``: (radius, y) from the bottom to the top;
    the first and last radius may be 0 (a pointed end is capped with a tiny disc)."""
    rings = []
    for r, y in profile:
        rings.append(circle((0.0, y, 0.0), max(r, 1e-4), (0.0, 1.0, 0.0), sections))
    return loft(rings)


def revolve_closed(polygon: list[tuple[float, float]], sections: int = 64) -> trimesh.Trimesh:
    """A closed (r, y) polygon turned around the Y axis: a ring, a hoop, a torus."""
    phis = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rings = [np.array([[r * np.cos(p), y, r * np.sin(p)] for r, y in polygon]) for p in phis]
    return loft(rings, closed=True)


def frames(path: np.ndarray):
    """Rotation-minimising frames along a polyline: tangents, normals, binormals."""
    path = np.asarray(path, np.float64)
    t = unit(np.gradient(path, axis=0))
    ref = np.array([0.0, 1.0, 0.0]) if abs(t[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = unit(np.cross(t[0], ref))
    normals = [n0]
    for i in range(1, len(path)):
        axis = np.cross(t[i - 1], t[i])
        s = np.linalg.norm(axis)
        n = normals[-1]
        if s > 1e-9:  # turn the previous normal with the tangent
            axis = axis / s
            ang = np.arctan2(s, t[i - 1] @ t[i])
            n = n * np.cos(ang) + np.cross(axis, n) * np.sin(ang) + axis * (axis @ n) * (1 - np.cos(ang))
        normals.append(unit(n - t[i] * (n @ t[i])))
    normals = np.array(normals)
    return t, normals, np.cross(t, normals)


def tube(path, radius, sections: int = 10) -> trimesh.Trimesh:
    """A round tube along a polyline; ``radius`` is a number or one value per path point."""
    path = np.asarray(path, np.float64)
    radii = np.broadcast_to(np.asarray(radius, np.float64), (len(path),))
    _, n, b = frames(path)
    t = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rings = [p + r * (np.cos(t)[:, None] * ni + np.sin(t)[:, None] * bi) for p, r, ni, bi in zip(path, radii, n, b)]
    return loft(rings)


def beam(path, width: float, height: float) -> trimesh.Trimesh:
    """A tube with a rectangular cross-section (a lath): ``width`` along the frame's normal,
    ``height`` along its binormal."""
    path = np.asarray(path, np.float64)
    _, n, b = frames(path)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    rings = [np.array([p + cx * width / 2 * ni + cy * height / 2 * bi for cx, cy in corners]) for p, ni, bi in zip(path, n, b)]
    return loft(rings)


def box(center, size) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=np.asarray(size, np.float64))
    m.apply_translation(np.asarray(center, np.float64))
    return trimesh.Trimesh(m.vertices, m.faces, process=False)


def cylinder(p0, p1, radius: float, sections: int = 20) -> trimesh.Trimesh:
    """A capped cylinder between two points."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    axis = p1 - p0
    return loft([circle(p0, radius, axis, sections), circle(p1, radius, axis, sections)])


def sphere(center, radius: float, sections: int = 32) -> trimesh.Trimesh:
    """UV sphere with its poles on the Y axis."""
    t = np.linspace(0, np.pi, sections // 2 + 1)[1:-1]
    profile = [(0.0, -radius)] + [(radius * np.sin(a), -radius * np.cos(a)) for a in t] + [(0.0, radius)]
    m = lathe(profile, sections)
    m.apply_translation(np.asarray(center, np.float64))
    return m


def rotate(mesh: trimesh.Trimesh, angle_deg: float, axis, origin=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle_deg), axis, origin))
    return m
