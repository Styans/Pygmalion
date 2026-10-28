"""Effects with drones of their own: where each drone of an effect is, and where it waits before and after.

An effect (a star that lights up when something is done, a ring spreading from a hoof, sparks, a splash, the
trail of a sword) is a group of drones apart from the figure. A drone show has a fixed fleet, and a drone cannot
appear out of nothing: the effect's drones wait dark at a **reserve** (a grid behind the figure, the safety
distance apart), fly to their places before the effect starts, light up, and fly back after it. Each waiting
place is matched to an effect place so that the flights are short and do not cross (the assignment problem,
``scipy.optimize.linear_sum_assignment``).

Shapes lie in the plane the audience sees (the figure's left and up), a ripple and a splash on the level (left and
forward, like a wave on water); all are centred on the effect's anchor;
the caller places them in the scene (the Blender add-on: around a point, the figure or a bone of it). Drones
of a shape are never closer than ``spacing``: when a shape is too small for the drones asked, it gets fewer
drones and says so.
"""

from __future__ import annotations

import numpy as np

from .errors import UserError

KINDS = {
    "star": "Звезда", "ring": "Кольцо", "heart": "Сердце", "ripple": "Расходящееся кольцо", "sparkles": "Искры",
    "splash": "Брызги", "points": "Свои точки (след, модель)",
}
MAX_DRONES = 2000
MIN_OUTLINE = 5  # fewer drones do not make a star, a ring or a heart one can tell
RIPPLE_START = 0.3  # a spreading ring starts at this share of its final size
ASSIGN_EXACT = 800  # the exact assignment is cubic: past this many drones, a sweep that keeps rows in order


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _closed(poly: np.ndarray) -> np.ndarray:
    return np.vstack([poly, poly[:1]])


def _length(poly: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def resample(poly: np.ndarray, n: int) -> np.ndarray:
    """``n`` points evenly along a polyline (closed when its ends meet)."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    closed = np.allclose(poly[0], poly[-1])
    at = np.linspace(0.0, arc[-1], n, endpoint=not closed)
    return np.stack([np.interp(at, arc, poly[:, k]) for k in range(poly.shape[1])], 1)


def outline(kind: str, size: float, params: dict) -> np.ndarray:
    """A closed outline (u: to the left, v: up) about ``size`` across, centred."""
    r = size / 2.0
    if kind == "star":
        arms = int(params.get("arms", 5))
        inner = float(params.get("inner", 0.45))
        a = np.pi / 2 + np.arange(2 * arms) * np.pi / arms
        rad = np.where(np.arange(2 * arms) % 2 == 0, r, r * inner)
        return _closed(np.stack([rad * np.cos(a), rad * np.sin(a)], 1))
    if kind in ("ring", "ripple"):
        a = np.linspace(0, 2 * np.pi, 361)
        return np.stack([r * np.cos(a), r * np.sin(a)], 1)
    if kind == "heart":
        t = np.linspace(0, 2 * np.pi, 721)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        pts = np.stack([x, y], 1)
        pts -= (pts.max(0) + pts.min(0)) / 2
        return pts * (size / np.ptp(pts, axis=0).max())
    raise UserError(f"Нет контура «{kind}»")


def closest(points: np.ndarray) -> float:
    """The smallest distance between two of the points."""
    p = np.asarray(points, np.float64)
    best = np.inf
    for a in range(0, len(p), 512):
        d = ((p[a: a + 512, None, :] - p[None, :, :]) ** 2).sum(-1)
        rows = np.arange(d.shape[0])
        d[rows, a + rows] = np.inf
        best = min(best, float(d.min()))
    return float(np.sqrt(best))


def even(poly: np.ndarray, spacing: float, count: int, what: str, notes: list) -> np.ndarray:
    """Points along an outline, no two closer than ``spacing`` in a straight line. The outline is cut finely and
    walked round, a point kept when no kept one is too close: by a sharp tip of a star or the dip of a heart a
    point or two drop out (the two sides come together there), elsewhere the drones go at the spacing."""
    length = _length(poly)
    _count(int(length / spacing), 3, what, [], least=MIN_OUTLINE)  # too small for a shape one can tell: say so
    step = spacing if count <= 0 else max(spacing, length / count)
    dense = resample(poly, max(3, int(length / step * 12)))
    pts = thin(dense, step, MAX_DRONES)
    if len(pts) < MIN_OUTLINE:  # the tips took the rest
        _count(len(pts), 0, what, [], least=MIN_OUTLINE)
    if count > len(pts):
        notes.append(f"{what}: при таком размере помещается {len(pts)} дронов на безопасном расстоянии, а не "
                     f"{count}: взято {len(pts)} (увеличьте размер, чтобы взять больше)")
    return pts[:count] if count > 0 else pts


def _count(capacity: int, count: int, what: str, notes: list, least: int = 1) -> int:
    if capacity < least:
        raise UserError(f"{what}: слишком мал для дронов на безопасном расстоянии — увеличьте размер")
    if count <= 0:
        return min(capacity, MAX_DRONES)
    if count > capacity:
        notes.append(f"{what}: при таком размере помещается {capacity} дронов на безопасном расстоянии, а не {count}: "
                     f"взято {capacity} (увеличьте размер, чтобы взять больше)")
        return capacity
    return min(count, MAX_DRONES)


def thin(points: np.ndarray, spacing: float, limit: int) -> np.ndarray:
    """Points in their order, each kept only if no kept one is closer than ``spacing``; at most ``limit``."""
    kept = np.empty((min(limit, len(points)), points.shape[1]))
    n = 0
    for p in points:
        if n >= limit:
            break
        if n == 0 or ((kept[:n] - p) ** 2).sum(1).min() >= spacing * spacing:
            kept[n] = p
            n += 1
    return kept[:n]


def scatter(n_max: int, radius: float, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Points in a ball, never closer than ``spacing`` (dart throwing)."""
    cand = rng.uniform(-1, 1, (40 * max(n_max, 8), 3))
    cand = cand[np.linalg.norm(cand, axis=1) <= 1.0] * radius
    return thin(cand, spacing, n_max)


def reserve(n: int, centre, left, up, spacing: float) -> np.ndarray:
    """A grid of ``n`` waiting places, ``spacing`` apart, in the plane facing the audience around ``centre``."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    k = np.arange(n)
    u = (k % cols - (cols - 1) / 2) * spacing
    v = (k // cols - (rows - 1) / 2) * spacing
    return np.asarray(centre, np.float64) + u[:, None] * _unit(left) + v[:, None] * _unit(up)


def _uncross(start: np.ndarray, target: np.ndarray, match: np.ndarray, left, up, rounds: int = 20) -> None:
    """Swap the starts of flights that cross as the audience sees them (the plane of left and up), until none do:
    each swap makes the flights shorter there, so it ends."""
    a2 = np.stack([start @ left, start @ up], 1)
    b2 = np.stack([target @ left, target @ up], 1)

    def side(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    for _ in range(rounds):
        swapped = False
        for i in range(len(match)):
            a, b = a2[match], b2
            cross = ((side(a[i], b[i], a) * side(a[i], b[i], b) < 0) & (side(a, b, a[i]) * side(a, b, b[i]) < 0))
            cross[i] = False
            js = np.nonzero(cross)[0]
            if len(js):
                j = int(js[0])
                match[i], match[j] = match[j], match[i]
                swapped = True
        if not swapped:
            return


def assign(start: np.ndarray, target: np.ndarray, left=(1, 0, 0), up=(0, 0, 1)) -> np.ndarray:
    """Which start goes to each target (an index per target): the shortest flights in all (by length, not its
    square: then flights do not cross), and any that still cross as the audience sees them are swapped."""
    n = len(target)
    if n <= ASSIGN_EXACT:
        from scipy.optimize import linear_sum_assignment

        cost = np.linalg.norm(target[:, None, :] - start[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        out = np.empty(n, np.int64)
        out[rows] = cols
        _uncross(start, target, out, _unit(left), _unit(up))
        return out
    order_t = np.lexsort((target[:, 0], target[:, 2]))  # rows bottom to top, left to right in each
    order_s = np.lexsort((start[:, 0], start[:, 2]))
    out = np.empty(n, np.int64)
    out[order_t] = order_s[:n]
    return out


def build(kind: str, size: float, spacing: float, frame: dict, reserve_at, count: int = 0, params: dict | None = None,
          points=None) -> dict:
    """Places of an effect's drones (``shape``, relative to its anchor, in the scene's axes), their waiting places
    (``reserve``) matched to them, and for a splash the starting ``velocity`` of each."""
    params = dict(params or {})
    if kind not in KINDS:
        raise UserError(f"Нет эффекта «{kind}». Есть: {', '.join(KINDS)}")
    if spacing <= 0 or size <= 0:
        raise UserError("Размер эффекта и расстояние между дронами должны быть больше нуля")
    left, up = _unit(frame["left"]), _unit(frame.get("up", (0, 0, 1)))
    forward = _unit(frame.get("forward", np.cross(left, up)))
    notes: list[str] = []
    rng = np.random.default_rng(int(params.get("seed", 0)))
    velocity = None
    title = KINDS[kind]
    if kind in ("star", "ring", "heart", "ripple"):
        poly = outline(kind, size if kind != "ripple" else size, params)
        if kind == "ripple":  # it starts as a ring of a third of its size and spreads to its whole size
            poly = poly * RIPPLE_START
        uv = even(poly, spacing, count, title, notes)
        n = len(uv)
        if kind == "ripple":
            start_r = float(np.linalg.norm(uv, axis=1).mean())
            params["expand"] = round(size / (2 * start_r) - 1.0, 4)
        # a ripple spreads on the level: rising and sinking drones are much slower than flying ones
        shape = uv[:, :1] * left + uv[:, 1:2] * (forward if kind == "ripple" else up)
    elif kind == "sparkles":
        shape = scatter(count if count > 0 else MAX_DRONES, size / 2, spacing, rng)
        n = _count(len(shape), count, title, notes)
        shape = shape[:n]
    elif kind == "splash":
        speed = float(params.get("speed", 4.0))  # m/s: slow enough for drones
        n = count if count > 0 else 30
        r0 = spacing * np.sqrt(n) / 1.6  # a disc they start from, the drones still apart
        uv = thin(np.concatenate([np.zeros((1, 2)), rng.uniform(-1, 1, (40 * n, 2)) * r0]), spacing, n)
        n = _count(len(uv), count, title, notes)
        uv = uv[:n]
        shape = uv[:, :1] * left + uv[:, 1:2] * forward
        out = shape / max(r0, 1e-9)
        velocity = speed * (out + up * float(params.get("lift", 1.2)))
        velocity /= np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True) / speed, 1e-9)
    else:  # points given by the caller: a trail along a sword's path, a model's surface
        given = np.asarray(points if points is not None else [], np.float64).reshape(-1, 3)
        if not len(given):
            raise UserError("Свои точки: пусто")
        keep = thin(given, spacing, MAX_DRONES)  # thinned to the safety distance, in their order
        if len(keep) < len(given):
            notes.append(f"Из {len(given)} точек оставлено {len(keep)}: остальные ближе безопасного расстояния")
        shape = keep[: _count(len(keep), count, title, notes)]
        n = len(shape)
    waiting = reserve(n, reserve_at, left, up, spacing)
    match = assign(waiting, shape, left, up)
    out = {"kind": kind, "count": int(n), "shape": shape.round(5).tolist(), "reserve": waiting[match].round(5).tolist(),
           "notes": notes, "params": params}
    if velocity is not None:
        out["velocity"] = velocity.round(5).tolist()
    return out
