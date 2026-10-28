"""Which branch of a skeleton tree is what: the spine with the head and the tail at its ends, legs, wings, arms;
front and back, left and right.

Works on any ``skeleton.Tree`` through a body (from voxel thinning or from a neural rig), so every way of
building a skeleton ends with the same named parts, and motions, effects and the chat can speak of
«the left wing» or «the hind legs» of any model.

1. Legs first: ends of the tree on the ground whose lower branch comes down steeply (with a body plan hint,
   the best ``n`` ends, so a flying horse with tucked legs still has four). Toes of one foot are one leg. A
   generated figure is usually mirror-symmetric (the plane x = c or y = c): an end on the ground lying on
   that plane with no mirror twin is a tail resting on the ground (a penguin, a horse), not a leg. Legs are
   paired by where their branches meet: a hip or a pair of shoulders.
2. Standing or lying: a standing body rises from the one place where its legs meet.
3. The spine is the thickest long path: the path with the most body around it (sum of radius² along it)
   between the ends that are neither feet nor wing tips (flat ends) and the places where legs meet. A
   lying body: from the muzzle to the tail tip; a standing one: from its hips up to the head, preferring
   ends above the hips to ends off to a side.
4. Everything hanging off the spine is a part; two big branches that part right next to the spine (both
   legs of one hip) are two parts. Parts with leg ends are legs; a thin plate standing out sideways is a
   wing (with its mirror twin, even if less flat); a limb sideways from the upper body of a standing figure
   without wings is an arm; a branch hanging down and back from its hips is its tail; the rest are
   ``extra`` (ears, horns, a crest): they still get bones, to move them and to hang effects on.
5. The head end of a lying spine: the neck leaves the body thick, a tail has a narrow root; then more body
   past the legs, the wings nearer the shoulders, and facing -Y on a tie (DroneGen imports models facing the
   Front view). The hips are where the hind legs meet the spine, the chest where the front legs, the arms
   or the wings do; past the chest is the neck and head, past the hips the tail.

Checked on ten generated models (horses, pegasi, a snow leopard, eagles, a penguin, a cartoon turtle): all
counts, standing or lying and facing right. Z is up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import UserError
from .skeleton import Tree, Volume

GROUND = 0.12  # an end this share of the height above the lowest point stands on the ground
LEG_STEEP = 0.5  # a leg goes down: -dz / length of its lower part at least this much
FLAT = 0.4  # thickness / width of a part below this: a plate (a wing, a fin)
SPLIT_NEAR = 0.2  # big branches parting within this share of a part's length from the spine are separate parts
SPLIT_BIG = 0.5  # ... when the smaller is at least this share of the larger
TINY = 0.06  # parts shorter than this share of the size get no bones (a bump, an ear of a small head)
UPRIGHT = 0.6  # the body above the legs this close to vertical (z of its direction): a standing figure
SIDEWAYS = 0.55  # a wing or an arm stands out sideways: its share across the spine at least this much
NECK = 2.0  # weight of «thicker where it leaves the body» (a neck, not the narrow root of a tail): the head end
MASS = 0.5  # weight of «more body past the legs» (a head and neck usually outweigh a tail)
WING_BONUS = 0.5  # weight of «the wings are nearer this end» when choosing the head end
TWIN = 0.35  # a mirror twin differs from its part by less than this (length share + place along the spine)
SHIN = 0.3  # a leg is judged by its branch up to this share of the height above the ground: the shin
SYMMETRIC = 0.6  # voxels matching their mirror image this well: a mirror-symmetric figure
CENTRAL = 0.6  # ... where an end on the ground this near the mirror plane (share of the farthest) is a tail, not a leg
FOOT = 0.25  # ends on the ground closer than this share of the height along the tree are toes of one foot

BODY_PLANS = {  # name -> hints; the Blender panel and the reference sheets use these names
    "auto": {},
    "quadruped": {"legs": 4, "wings": 0, "arms": 0, "upright": False},
    "winged_quadruped": {"legs": 4, "wings": 2, "arms": 0, "upright": False},
    "bird": {"legs": 2, "wings": 2, "arms": 0},
    "biped": {"legs": 2, "arms": 2, "upright": True},
    "winged_biped": {"legs": 2, "arms": 2, "wings": 2, "upright": True},
    "snake": {"legs": 0, "wings": 0, "arms": 0},
    "rigid": {"rigid": True},
}


@dataclass
class Part:
    role: str  # leg_front | leg_hind | leg | arm | wing | extra
    chain: list[int]  # tree nodes from the spine node it grows from to its tip
    nodes: set[int]  # every node of the part, its twigs too (not the spine node)
    attach: int  # index into Body.spine where it grows
    length: float
    flatness: float = 1.0
    volume: float = 0.0
    centroid: np.ndarray = field(default=None, repr=False)
    side: str = ""  # L | R | ""
    number: int = 0  # 1, 2... when a role and side has several parts

    @property
    def tip(self) -> int:
        return self.chain[-1]


@dataclass
class Body:
    spine: list[int]  # tree nodes from the tail end to the head end
    hips: int  # index into spine: where the hind legs (the legs of a standing figure) grow; before it, the tail
    chest: int  # index into spine: where the front legs, arms or wings grow; after it, the neck and head
    parts: list[Part]
    upright: bool
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray
    ground: float
    height: float
    size: float
    rigid: bool = False
    notes: list[str] = field(default_factory=list)

    def of(self, role: str) -> list[Part]:
        return [p for p in self.parts if p.role == role]


def hints_of(plan) -> dict:
    """Hints from a body plan name, a dict, or None (automatic)."""
    if plan is None or plan == "":
        return {}
    if isinstance(plan, str):
        if plan not in BODY_PLANS:
            raise UserError(f"Неизвестный план тела «{plan}»: {', '.join(BODY_PLANS)}")
        return dict(BODY_PLANS[plan])
    return dict(plan)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def _walk(tree: Tree, start: int, allowed=None, weight=None) -> tuple[dict, dict]:
    """(previous node, distance) from ``start`` over the nodes in ``allowed`` (all when None).
    ``weight(i, j)`` is the cost of a step; its length by default."""
    prev, dist, order = {start: -1}, {start: 0.0}, [start]
    for i in order:
        for j in tree.nbrs[i]:
            if j not in prev and (allowed is None or j in allowed):
                prev[j] = i
                step = float(np.linalg.norm(tree.points[i] - tree.points[j]))
                dist[j] = dist[i] + (weight(i, j, step) if weight else step)
                order.append(j)
    return prev, dist


def _back(prev: dict, end: int) -> list[int]:
    path = [end]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])
    return path[::-1]


def _subtree(tree: Tree, start: int, blocked: set[int]) -> list[int]:
    out, seen = [start], {start} | blocked
    for i in out:
        for j in tree.nbrs[i]:
            if j not in seen:
                seen.add(j)
                out.append(j)
    return out


def _main_chain(tree: Tree, base: int, nodes: set[int]) -> list[int]:
    """From ``base`` to the tip of a part: the farthest end, but a thick end (a muzzle) wins over a thin one (an ear)
    of nearly the same reach."""
    prev, dist = _walk(tree, base, nodes | {base})
    ends = [i for i in prev if i != base and all(j == prev[i] or j not in prev for j in tree.nbrs[i])]
    if not ends:
        return [base]
    thick = max(tree.radius[e] for e in ends) or 1.0
    tip = max(ends, key=lambda e: dist[e] * (max(tree.radius[e], 1e-9) / thick) ** 0.25)
    return _back(prev, tip)


def _split(tree: Tree, base: int, nodes: list[int]) -> list[list[int]]:
    """A branch off the spine holding two big parts that part next to it (both legs of one hip) -> each part
    with the short common stem."""
    nodes_set = set(nodes)
    chain = _main_chain(tree, base, nodes_set)
    full = tree.length(chain)
    stem: list[int] = []
    for k, i in enumerate(chain[1:], 1):
        if tree.length(chain[: k + 1]) > SPLIT_NEAR * full:
            break
        stem.append(i)
        children = [j for j in tree.nbrs[i] if j in nodes_set and j not in chain[:k]]
        if len(children) < 2:
            continue
        blocked = set(chain[: k + 1])
        subs = [_subtree(tree, j, blocked) for j in children]
        reach = [tree.length(_main_chain(tree, i, set(s))) for s in subs]
        big = [s for s, r in zip(subs, reach) if r >= SPLIT_BIG * max(reach)]
        if len(big) < 2:
            continue
        rest = [x for s, r in zip(subs, reach) if r < SPLIT_BIG * max(reach) for x in s]
        big[int(np.argmax([r for r in reach if r >= SPLIT_BIG * max(reach)]))] += rest
        out = []
        for s in big:
            out.extend(_split(tree, base, stem + s))
        return out
    return [nodes]


def _owners(tree: Tree) -> dict[int, np.ndarray]:
    """Node -> indices of the filled voxels it owns (``Tree.owner``)."""
    if tree.owner is None:
        return {}
    order = np.argsort(tree.owner, kind="stable")
    bounds = np.searchsorted(tree.owner[order], np.arange(len(tree) + 1))
    return {n: order[bounds[n]: bounds[n + 1]] for n in range(len(tree)) if bounds[n + 1] > bounds[n]}


def _flatness(tree: Tree, chain: list[int], cells: np.ndarray, owned: dict[int, np.ndarray], pitch: float) -> float:
    """How thick a stretch of the tree is (its radius) against how far its voxels spread from it: a cylinder
    spreads as far as it is thick (~1), a plate much farther (a wing: thickness / chord)."""
    ratios = [tree.radius[n] / max(np.linalg.norm(cells[owned[n]] - tree.points[n], axis=1).max(), pitch)
              for n in chain if n in owned]
    return float(np.median(ratios)) if ratios else 1.0


def _features(part: Part, tree: Tree, cells: np.ndarray, owned: dict[int, np.ndarray], pitch: float) -> None:
    own = [owned[n] for n in part.nodes if n in owned]
    pts = cells[np.concatenate(own)] if own else tree.points[list(part.nodes)]
    part.volume = len(pts) * pitch**3
    part.centroid = pts.mean(axis=0)
    # the inner fifth is left out: there the part's voxels mix with the shoulder it grows from
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(tree.points[part.chain], axis=0), axis=1))])
    part.flatness = _flatness(tree, [n for n, s in zip(part.chain, arc) if s >= 0.2 * arc[-1]], cells, owned, pitch)


def label(tree: Tree, vol: Volume, hints=None) -> Body:
    """Name the parts of a skeleton tree."""
    hints = hints_of(hints)
    cells = vol.cells()
    lo, hi = cells.min(axis=0), cells.max(axis=0)
    ground, height, size = float(lo[2]), float(hi[2] - lo[2]) or vol.pitch, float((hi - lo).max()) or vol.pitch
    up = np.array([0.0, 0.0, 1.0])
    pts, rad = tree.points, tree.radius
    core = int(np.argmax(rad))
    if hints.get("rigid") or len(tree) < 3:
        return _rigid(tree, core, ground, height, size, up, hints)

    # 1. legs: ends on the ground, reached going down; the toes of one foot are one leg
    parent, _ = tree.rooted(core)
    parent_of = {i: int(parent[i]) for i in range(len(tree))}
    ends = tree.ends()
    to_core = {e: _back(parent_of, e)[::-1] for e in ends}  # from the end towards the core
    owned = _owners(tree)

    def steep(e: int) -> float:  # how straight down the lower part of the branch comes: a shin, not a dragged tail
        path = to_core[e]
        k = next((i for i, x in enumerate(path) if pts[x][2] >= ground + SHIN * height), len(path) - 1)
        if len(path) < 2:
            return 0.0
        d = pts[e] - pts[path[max(k, 1)]]
        return float(-d[2] / max(np.linalg.norm(d), 1e-12))

    def grounded(e: int) -> bool:
        return pts[e][2] - rad[e] - ground < GROUND * height

    # legs of a mirror-symmetric figure stand to the sides of its mirror plane; a tail resting on the ground (a
    # penguin, a horse) lies on the plane
    normal, plane, match = _mirror_plane(vol)
    symmetric = match >= SYMMETRIC

    def lower(e: int) -> list[int]:  # the branch of an end up to the shin height
        path = to_core[e]
        return path[: next((i for i, x in enumerate(path) if pts[x][2] >= ground + SHIN * height), len(path)) or 1]

    def off(f: list[int]) -> float:  # by the middle of the lower branch: a tail fan ends in a corner, but lies on the plane
        return abs(float(pts[lower(f[0])].mean(axis=0) @ normal) - plane)

    feet = _feet(tree, parent, sorted(ends, key=lambda e: steep(e) + grounded(e), reverse=True), FOOT * height)
    standing = [f for f in feet if grounded(f[0])]
    spread = max([off(f) for f in standing] or [0.0])  # how far the feet on the ground stand from the mirror plane

    def sided(f: list[int]) -> bool:
        """Not a tail on the mirror plane: off the plane, or with a mirror twin among the other feet."""
        if not symmetric or off(f) >= CENTRAL * spread:
            return True
        p = pts[f[0]]
        mirrored = p - 2 * (float(p @ normal) - plane) * normal
        return any(g is not f and np.linalg.norm(pts[g[0]] - mirrored) < TWIN * height for g in standing)

    if hints.get("legs") is not None:
        leg_feet = sorted(feet, key=lambda f: steep(f[0]) + grounded(f[0]) + 0.5 * (grounded(f[0]) and sided(f)),
                          reverse=True)[: hints["legs"]]
    else:
        leg_feet = [f for f in standing if steep(f[0]) >= LEG_STEEP]
        if len([f for f in leg_feet if sided(f)]) >= 2:
            leg_feet = [f for f in leg_feet if sided(f)]
    leg_ends = [max(f, key=lambda e: tree.length(to_core[e])) for f in leg_feet]  # the longest toe is the leg's tip
    toes = {e for f in leg_feet for e in f}
    groups = _pair_legs(tree, parent, leg_ends)

    # 2. standing or lying: a standing body rises from where its legs meet
    upright = hints.get("upright")
    if upright is None and groups:
        upright = len(groups) == 1 and bool(_unit(pts[core] - pts[groups[0]["meet"]])[2] >= UPRIGHT)

    # 3. the spine: the path with the most body around it; wing tips (flat ends) do not end it
    flat = {e for e in ends if e not in toes and _flatness(tree, to_core[e][: max(2, len(to_core[e]) // 2)],
                                                           cells, owned, vol.pitch) < FLAT}
    anchors = [e for e in ends if e not in toes and e not in flat] or [e for e in ends if e not in toes]
    if upright and groups:  # from the hips up to the head; a tail of a standing figure is a part
        meet = groups[0]["meet"]
        side = normal if symmetric else _unit(np.cross(up, _toes_forward(tree, [to_core[e][::-1] for e in leg_ends], up)))

        def central(b: int) -> float:  # the head is above the hips, not off to a side like a wing tip or a hand
            return max(0.05, 1.0 - abs(float(_unit(pts[b] - pts[meet]) @ side))) ** 2

        spine = _thickest_path(tree, anchors, start=meet, rank=central)
    else:
        spine = _thickest_path(tree, sorted(set(anchors) | {g["meet"] for g in groups}))
    if spine is None:
        prev, dist = _walk(tree, anchors[0] if anchors else core)
        spine = _back(prev, max(dist, key=dist.get))
    on_spine = set(spine)

    # 4. parts hanging off it
    parts: list[Part] = []
    for k, s in enumerate(spine):
        for j in tree.nbrs[s]:
            if j in on_spine:
                continue
            for nodes in _split(tree, s, _subtree(tree, j, on_spine)):
                chain = _main_chain(tree, s, set(nodes))
                part = Part("extra", chain, set(nodes), k, tree.length(chain))
                if part.length >= TINY * size or any(e in part.nodes for e in leg_ends):
                    _features(part, tree, cells, owned, vol.pitch)
                    parts.append(part)
    legs = [p for p in parts if any(e in p.nodes for e in leg_ends)]
    for p in legs:
        p.role = "leg"
    n = len(spine) - 1
    hips_at = sorted({int(round(np.mean([p.attach for p in legs if p.nodes & set(g["ends"])])))
                      for g in groups if any(p.nodes & set(g["ends"]) for p in legs)})
    axis = _unit(pts[spine[-1]] - pts[spine[0]]) if n > 0 else up
    if upright is None:
        upright = bool(abs(axis[2]) >= UPRIGHT)

    # left and right before the head end is known (the sign does not matter yet): across the mirror plane, across
    # the toes of standing feet, across the spine of a lying body
    if symmetric:
        left = normal
    elif upright:
        left = _unit(np.cross(up, _toes_forward(tree, [p.chain for p in legs], up)))
    else:
        flat_axis = axis - (axis @ up) * up
        left = _unit(np.cross(up, flat_axis)) if np.linalg.norm(flat_axis) > 1e-6 else np.array([1.0, 0.0, 0.0])

    def across(p: Part) -> float:  # share of its direction to the left or right
        d = pts[p.tip] - pts[spine[p.attach]]
        return abs(float(d @ left)) / max(np.linalg.norm(d), 1e-12)

    free = [p for p in parts if p.role == "extra"]

    def take(role: str, score, count: int | None, ok, single: bool = False) -> list[Part]:
        """With a hint the best ``count`` parts whatever they look like (the user knows better); else the plausible
        (the best one of them when ``single``), each with its mirror twin: a part of about the same length on the
        other side (one wing of a pair may be drawn less flat than the other)."""
        ranked_parts = sorted((p for p in free if count is not None or ok(p)), key=score, reverse=True)
        chosen = ranked_parts[:count] if count is not None else ranked_parts[:1] if single else ranked_parts
        if count is None and not single:
            for p in list(chosen):
                twin = _twin(p, [q for q in free if q not in chosen], pts, spine, left)
                if twin is not None:
                    chosen.append(twin)
        for p in chosen:
            p.role = role
            free.remove(p)
        return chosen

    wings = take("wing", lambda p: (1 - p.flatness) + across(p) + p.length / size, hints.get("wings"),
                 lambda p: p.flatness < FLAT and across(p) >= SIDEWAYS and p.length >= 0.25 * size)

    # 5. the head end: tail -> head along the spine
    if n > 0 and _head_first(tree, spine, parts, wings, hips_at, owned, vol.pitch, upright, bool(groups)):
        spine.reverse()
        for p in parts:
            p.attach = n - p.attach
        hips_at = sorted(n - a for a in hips_at)
        axis = -axis

    arms, tails = [], []
    if upright:
        lowest = hips_at[0] if hips_at else 0

        def shoulder(p: Part) -> bool:  # arms grow from the upper body, below the neck (not a braid, not a bandana)
            return 0.25 <= (p.attach - lowest) / max(n - lowest, 1) <= 0.88

        # an arm is a limb, not a ribbon or a feather; a winged figure (a bird) has arms only when its plan says so
        arms = take("arm", lambda p: across(p) + 2 * p.length / size - 2 * (not shoulder(p)) - (p.flatness < FLAT),
                    hints.get("arms"), lambda p: not wings and across(p) >= SIDEWAYS and shoulder(p)
                    and p.flatness >= FLAT and p.length >= 0.15 * size)
        if hints.get("tail") is not False:
            tails = take("tail", lambda p: p.length, 1 if hints.get("tail") else None,
                         lambda p: p.attach <= lowest + 0.3 * (n - lowest) and across(p) < 0.5 and p.length >= 0.12 * size
                         and pts[p.tip][2] < pts[spine[p.attach]][2] + 0.1 * height, single=True)
    elif hints.get("arms"):
        arms = take("arm", lambda p: across(p) + p.attach / max(n, 1), hints["arms"], None)

    hips = hips_at[0] if hips_at else 0
    if len(hips_at) >= 2:
        chest = hips_at[-1]
    elif arms or wings:
        chest = int(round(np.mean([p.attach for p in (arms or wings)])))
    else:
        chest = hips + int(round(0.75 * (n - hips)))
    if chest - hips < 2 and n >= 2:
        chest = min(n, hips + max(2, int(round(0.75 * (n - hips)))))
    for p in legs:
        if len(groups) >= 2:
            p.role = "leg_hind" if abs(p.attach - hips) <= abs(p.attach - chest) else "leg_front"

    body = Body(spine, hips, chest, parts, bool(upright), np.zeros(3), np.zeros(3), up, ground, height, size)
    _frame(body, tree, legs, normal if symmetric else None)
    _sides(body, tree)
    _notes(body, hints, leg_ends)
    return body


def _mirror_plane(vol: Volume) -> tuple[np.ndarray, float, float]:
    """The left/right mirror plane of a figure among the planes x = c and y = c (DroneGen models face -Y, or ±X
    when drawn from the side): (its normal, its offset along the normal, how well the voxels match their mirror
    image across it, 0..1)."""
    from ..postprocess import _mirrored, symmetry_plane

    best = (np.array([1.0, 0.0, 0.0]), 0.0, 0.0)
    for axis in (0, 1):
        g = np.moveaxis(vol.grid, axis, 0)
        c2 = symmetry_plane(g)
        mirror = _mirrored(g, c2)
        match = float((g & mirror).sum() / max((g | mirror).sum(), 1))
        if match > best[2]:
            normal = np.zeros(3)
            normal[axis] = 1.0
            best = (normal, float(vol.origin[axis] + c2 / 2 * vol.pitch), match)
    return best


def _lineage(parent: np.ndarray, e: int) -> list[int]:
    out = [e]
    while parent[out[-1]] >= 0:
        out.append(int(parent[out[-1]]))
    return out


def _apart(tree: Tree, lines: dict, a: int, b: int) -> tuple[int, float]:
    """(where the branches of two ends meet, the length of the path between the ends)."""
    on_b = set(lines[b])
    m = next(x for x in lines[a] if x in on_b)
    return m, tree.length(lines[a][: lines[a].index(m) + 1]) + tree.length(lines[b][: lines[b].index(m) + 1])


def _feet(tree: Tree, parent: np.ndarray, ranked: list[int], reach: float) -> list[list[int]]:
    """Ends grouped into feet: toes closer than ``reach`` along the tree are one foot. In the order of ``ranked``."""
    lines = {e: _lineage(parent, e) for e in ranked}
    feet: list[list[int]] = []
    for e in ranked:
        for foot in feet:
            if _apart(tree, lines, e, foot[0])[1] < reach:
                foot.append(e)
                break
        else:
            feet.append([e])
    return feet


def _twin(part: Part, others: list[Part], pts: np.ndarray, spine: list[int], left: np.ndarray) -> Part | None:
    """The part mirroring ``part`` across the body: on the other side, about as long, growing near the same place."""
    side = float((pts[part.tip] - pts[spine[part.attach]]) @ left)
    best, best_diff = None, TWIN
    for q in others:
        if float((pts[q.tip] - pts[spine[q.attach]]) @ left) * side >= 0:
            continue
        diff = abs(q.length - part.length) / max(part.length, 1e-9) + abs(q.attach - part.attach) / max(len(spine), 1)
        if diff < best_diff:
            best, best_diff = q, diff
    return best


def _pair_legs(tree: Tree, parent: np.ndarray, leg_ends: list[int]) -> list[dict]:
    """Legs in pairs by where their branches meet (both front legs at the shoulders), nearest meeting first."""
    lines = {e: _lineage(parent, e) for e in leg_ends}
    pairs = sorted(((_apart(tree, lines, a, b), a, b) for i, a in enumerate(leg_ends) for b in leg_ends[i + 1:]),
                   key=lambda x: x[0][1])
    groups, used = [], set()
    for (m, _), a, b in pairs:
        if a not in used and b not in used:
            groups.append({"ends": [a, b], "meet": m})
            used |= {a, b}
    for e in leg_ends:
        if e not in used:  # an odd leg: it meets the body where its own branch joins a bigger one
            m = next((x for x in lines[e][1:] if len(tree.nbrs[x]) >= 3), lines[e][-1])
            groups.append({"ends": [e], "meet": m})
    return groups


def _thickest_path(tree: Tree, anchors: list[int], start: int | None = None, rank=None) -> list[int] | None:
    """The path between two anchors (or from ``start`` to an anchor) with the largest sum of radius² along it,
    times ``rank(anchor)`` when given."""
    rad = tree.radius

    def weight(i, j, step):
        return step * 0.5 * (rad[i] ** 2 + rad[j] ** 2)

    best, best_path = -1.0, None
    for a in ([start] if start is not None else anchors):
        prev, dist = _walk(tree, a, weight=weight)
        for b in anchors:
            value = dist.get(b, -1.0) * (rank(b) if rank else 1.0)
            if b != a and value > best:
                best, best_path = value, (prev, b)
    return _back(*best_path) if best_path else None


def _head_first(tree: Tree, spine: list[int], parts: list[Part], wings: list[Part], hips_at: list[int],
                owned: dict, pitch: float, upright: bool, from_hips: bool) -> bool:
    """Whether the spine runs head -> tail now (it should run tail -> head).

    A lying body: the neck leaves the body thick, a tail has a narrow root (measured on six horses, a pegasus
    and a snow leopard whose fluffy tail outweighs its head; the root told every one right). Then more body
    past the legs, the wings nearer the shoulders, and facing -Y on a tie.
    """
    pts, n = tree.points, len(spine) - 1
    if upright:
        if from_hips:  # built from where the legs meet up to the head
            return False
        return pts[spine[0]][2] > pts[spine[-1]][2]
    lo, hi = (hips_at[0], hips_at[-1]) if hips_at else (n // 2, n // 2)
    segments = {0: spine[lo::-1], n: spine[hi:]}  # from the legs outwards

    def root(end: int) -> float:
        r = tree.radius[segments[end]]
        return float(r[: max(2, len(r) // 4)].mean())

    def mass(end: int) -> float:
        on = set(segments[end])
        vol = sum(len(owned.get(s, ())) for s in on) * pitch**3
        return vol + sum(p.volume for p in parts if p.role == "extra" and spine[p.attach] in on)

    roots, masses = {e: root(e) for e in (0, n)}, {e: mass(e) for e in (0, n)}
    score = {}
    for end in (0, n):
        score[end] = (NECK * roots[end] / max(roots[0] + roots[n], 1e-12)
                      + MASS * masses[end] / max(masses[0] + masses[n], 1e-12))
        if wings:
            mean_attach = np.mean([p.attach for p in wings])
            score[end] += WING_BONUS * (1.0 - abs(mean_attach - end) / max(n, 1))
        facing = _unit(pts[spine[end]] - pts[spine[n - end]])
        score[end] += 0.1 * float(facing @ np.array([0.0, -1.0, 0.0]))
    return score[0] > score[n]


def _toes_forward(tree: Tree, legs: list[list[int]], up: np.ndarray) -> np.ndarray:
    """Where standing feet point (the toes), or -Y when they do not tell. ``legs``: chains from the body to the toes."""
    pts = tree.points
    forward, reach = np.zeros(3), 0.0
    for chain in legs:
        c = pts[chain]
        toe = c[-1] - c[max(0, int(0.8 * len(c)) - 1)]
        forward += toe - (toe @ up) * up
        reach += tree.length(chain)
    if not legs or np.linalg.norm(forward) < 0.05 * reach:
        return np.array([0.0, -1.0, 0.0])
    return _unit(forward)


def _frame(body: Body, tree: Tree, legs: list[Part], mirror: np.ndarray | None) -> None:
    """Forward: tail -> head of a lying body; for a standing one, across its mirror plane (when it has one) on the
    side the toes point to, -Y when they do not tell."""
    pts, up = tree.points, body.up
    forward = np.zeros(3)
    if not body.upright and len(body.spine) > 1:
        forward = pts[body.spine[body.chest]] - pts[body.spine[body.hips]]
        if np.linalg.norm(forward - (forward @ up) * up) < 1e-6:
            forward = pts[body.spine[-1]] - pts[body.spine[0]]
        forward = forward - (forward @ up) * up
    if np.linalg.norm(forward) < 1e-9:
        toes = _toes_forward(tree, [p.chain for p in legs], up)
        if mirror is not None:
            across = _unit(np.cross(mirror, up))
            forward = across if across @ toes >= 0 else -across
        else:
            forward = toes
    body.forward = _unit(forward)
    body.left = _unit(np.cross(up, body.forward))


def _sides(body: Body, tree: Tree) -> None:
    pts = tree.points
    groups: dict[str, list[Part]] = {}
    for p in body.parts:
        groups.setdefault(p.role, []).append(p)
    for role, group in groups.items():
        offset = {id(p): float((p.centroid - pts[body.spine[p.attach]]) @ body.left) for p in group}
        if role == "extra":
            for p in group:
                p.side = "" if abs(offset[id(p)]) < 0.08 * body.size else ("L" if offset[id(p)] > 0 else "R")
        else:
            ranked = sorted(group, key=lambda p: offset[id(p)], reverse=True)
            o1, o2 = (offset[id(p)] for p in ranked[:2]) if len(ranked) == 2 else (0.0, 0.0)
            one_side = o1 * o2 > 0 and min(abs(o1), abs(o2)) > 0.1 * body.size
            if len(ranked) == 2 and not one_side:  # a pair is one left and one right even when both lean to one side
                ranked[0].side, ranked[1].side = "L", "R"
            else:
                for p in ranked:
                    o = offset[id(p)]
                    p.side = "L" if o > 0.02 * body.size else "R" if o < -0.02 * body.size else ""
        counts: dict[tuple[str, str], int] = {}
        for p in sorted(group, key=lambda p: p.attach):
            key = (p.role, p.side)
            counts[key] = counts.get(key, 0) + 1
            p.number = counts[key]
        for p in group:
            if counts[(p.role, p.side)] == 1:
                p.number = 0


WORDS = {"legs": "ноги", "wings": "крылья", "arms": "руки"}


def _notes(body: Body, hints: dict, leg_ends: list[int]) -> None:
    found = {"legs": sum(1 for p in body.parts if p.role.startswith("leg")), "wings": len(body.of("wing")),
             "arms": len(body.of("arm"))}
    for key, word in WORDS.items():
        want = hints.get(key)
        if want is not None and found[key] < want:
            body.notes.append(f"{word.capitalize()}: нашёл {found[key]} из {want} — возможно, слиплись с телом или "
                              f"друг с другом; поправьте скелет вручную или сгенерируйте модель в позе «Для анимации»")
    if len(leg_ends) > found["legs"]:
        body.notes.append("Часть ног не отделилась от тела")


def _rigid(tree: Tree, core: int, ground: float, height: float, size: float, up: np.ndarray, hints: dict) -> Body:
    """An object without limbs (a dombra, a yurt): the longest path is its only chain."""
    if len(tree) > 1:
        prev, dist = _walk(tree, core)
        prev, dist = _walk(tree, max(dist, key=dist.get))
        spine = _back(prev, max(dist, key=dist.get))
    else:
        spine = [core]
    pts = tree.points
    if len(spine) > 1 and pts[spine[0]][2] > pts[spine[-1]][2]:
        spine.reverse()
    return Body(spine, 0, len(spine) - 1, [], True, np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), up,
                ground, height, size, rigid=bool(hints.get("rigid")))
