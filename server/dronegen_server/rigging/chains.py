"""Named parts of a body -> bones: each part's chain is smoothed and cut into joints at fixed shares of its length.

Names follow Blender's convention, so mirroring, pose flipping and «select mirror» work: ``leg_front_2.L``,
``wing_1.R``, ``tail_3``, ``neck_1``, ``head``. Every bone also carries its role, side, chain and place in the
chain, which is all that motions and effects need: they never look at names.

A ``root`` bone lies on the ground under the hips: moving it moves the whole figure (flight, a jump).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .roles import Body, Part
from .skeleton import Tree

SHARES = {  # share of the chain's length taken by each bone, from the body out
    "leg4": (0.3, 0.3, 0.28, 0.12),  # four-legged: thigh, shin, cannon, hoof / paw
    "leg2": (0.45, 0.42, 0.13),  # standing: thigh, shin, foot
    "arm": (0.45, 0.4, 0.15),  # upper arm, forearm, hand
    "wing": (0.3, 0.33, 0.37),  # arm, forearm, hand of a wing
    "neck_head_upright": (0.3, 0.7),
    "neck_head": (0.3, 0.3, 0.4),
}
TAIL_BONE = 0.07  # a tail gets a bone per this share of the size (2 to 8 bones)
EXTRA_BONE = 0.1
TWIG = 0.12  # a side branch of a part this long (share of the size) gets its own chain: antlers, a crest, a sword
TWIG_FROM = 0.4  # ... when it grows on the outer part of a limb (past this share of its length)
PROP_RATIO = 1.35  # an arm this much longer than its pair holds something: the rest past the hand is a ``prop``


@dataclass
class Bone:
    name: str
    role: str
    side: str
    chain: str  # e.g. "leg_front.L": every bone of one chain
    segment: int  # 1, 2... from the body out
    head: list
    tail: list
    parent: str | None
    deform: bool = True


def _smooth(points: np.ndarray, window: int = 5) -> np.ndarray:
    """Voxel steps out of a polyline; the ends stay where they are."""
    if len(points) <= 2:
        return points
    k = min(window, len(points) if len(points) % 2 else len(points) - 1)
    pad = k // 2
    padded = np.concatenate([np.repeat(points[:1], pad, 0), points, np.repeat(points[-1:], pad, 0)])
    kernel = np.ones(k) / k
    out = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)], 1)
    out[0], out[-1] = points[0], points[-1]
    return out


def _arc(points: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _at(points: np.ndarray, arc: np.ndarray, s: float) -> np.ndarray:
    return np.array([np.interp(s, arc, points[:, i]) for i in range(3)])


class _Builder:
    def __init__(self, tree: Tree, body: Body):
        self.tree, self.body = tree, body
        self.bones: list[Bone] = []
        self.ranges: dict[str, tuple[np.ndarray, list[str]]] = {}  # chain -> (joint arc positions, bone names)
        self.nodes: dict[str, np.ndarray] = {}  # chain -> arc position of each tree node of the chain
        self.node_ids: dict[str, list[int]] = {}

    def chain(self, name: str, role: str, side: str, nodes: list[int], shares, parent: str | None,
              names: list[str] | None = None, start: np.ndarray | None = None) -> list[str]:
        pts = self.tree.points[nodes]
        if start is not None:
            pts = np.concatenate([start[None], pts])
        pts = _smooth(pts)
        arc = _arc(pts)
        total = arc[-1]
        if total <= 1e-9:
            return []
        shares = np.asarray(shares, np.float64) / np.sum(shares)
        cuts = np.concatenate([[0.0], np.cumsum(shares)]) * total
        joints = [_at(pts, arc, s) for s in cuts]
        suffix = f".{side}" if side else ""
        if names is None:
            names = [f"{name}_{k}" for k in range(1, len(shares) + 1)] if len(shares) > 1 else [name]
        out = []
        for k, bone_name in enumerate(names):
            self.bones.append(Bone(bone_name + suffix, role, side, name + suffix, k + 1, joints[k].tolist(),
                                   joints[k + 1].tolist(), parent))
            parent = bone_name + suffix
            out.append(parent)
        key = name + suffix
        self.ranges[key] = (cuts, out)
        self.nodes[key] = arc[1:] if start is not None else arc
        self.node_ids[key] = list(nodes)
        return out

    def bone_at(self, chain: str, node: int) -> str:
        """The bone of a chain that holds a tree node of it."""
        cuts, names = self.ranges[chain]
        ids = self.node_ids[chain]
        s = self.nodes[chain][ids.index(node)] if node in ids else 0.0
        return names[int(np.clip(np.searchsorted(cuts, s, side="right") - 1, 0, len(names) - 1))]


def bones(tree: Tree, body: Body) -> list[Bone]:
    b = _Builder(tree, body)
    pts, size = tree.points, body.size
    spine = body.spine
    hips_pt = pts[spine[body.hips]]
    root_head = np.array([hips_pt[0], hips_pt[1], body.ground])
    b.bones.append(Bone("root", "root", "", "root", 1, root_head.tolist(),
                        (root_head + body.forward * 0.15 * size).tolist(), None, deform=False))

    if body.rigid or not body.parts:
        nodes = spine
        if tree.length(nodes) < 0.05 * size:  # a lump: one upright bone through its middle
            c = pts[spine[0]]
            b.bones.append(Bone("body", "spine", "", "body", 1, (c - [0, 0, 0.25 * size]).tolist(),
                                (c + [0, 0, 0.25 * size]).tolist(), "root"))
        else:
            n = int(np.clip(round(tree.length(nodes) / (0.15 * size)), 1, 6))
            b.chain("body", "spine", "", nodes, [1] * n, "root")
        return b.bones

    # the spine: hips -> chest; a long limbless body (a snake) gets more bones
    torso = spine[body.hips: body.chest + 1]
    limbless = not any(p.role.startswith(("leg", "arm", "wing")) for p in body.parts)
    n = int(np.clip(round(tree.length(torso) / (0.1 * size)), 3, 10)) if limbless else 3
    spine_bones = b.chain("spine", "spine", "", torso, [1] * n, "root")

    # past the chest: the neck and the head; before the hips: the tail
    neck = spine[body.chest:]
    if len(neck) > 1 and tree.length(neck) >= 0.05 * size:
        shares = SHARES["neck_head_upright" if body.upright else "neck_head"]
        names = ["neck_1", "head"] if len(shares) == 2 else ["neck_1", "neck_2", "head"]
        b.chain("neck", "neck", "", neck, shares, spine_bones[-1], names=names)
        for bone in b.bones:  # the last bone is the head itself
            if bone.name == "head":
                bone.role = "head"
    tail = spine[: body.hips + 1][::-1]
    if len(tail) > 1 and tree.length(tail) >= 0.05 * size:
        count = int(np.clip(round(tree.length(tail) / (TAIL_BONE * size)), 2, 8))
        b.chain("tail", "tail", "", tail, [1] * count, spine_bones[0])

    def spine_parent(p: Part) -> str:
        """Limbs hang from the body itself (hind legs from the hips, the rest from the chest), even when they grow a
        little past it; small parts from whatever bone they grow on (ears from the head, a crest from the neck)."""
        if p.role != "extra":
            if p.attach <= body.hips or p.role == "leg_hind":
                return spine_bones[0]
            if p.attach >= body.chest:
                return spine_bones[-1]
            return b.bone_at("spine", spine[p.attach])
        node = spine[p.attach]
        for key in ("spine", "neck", "tail"):
            if key in b.node_ids and node in b.node_ids[key] and (key == "spine" or node != b.node_ids[key][0]):
                return b.bone_at(key, node)
        return spine_bones[0] if p.attach <= body.hips else spine_bones[-1]

    arms = body.of("arm")
    pair_len = {p.side: p.length for p in arms}
    for p in sorted(body.parts, key=lambda p: (p.role, p.side, p.number)):
        name = p.role + (str(p.number) if p.number else "")
        parent = spine_parent(p)
        if p.role.startswith("leg"):
            b.chain(name, p.role, p.side, p.chain, SHARES["leg4" if p.role != "leg" else "leg2"], parent)
        elif p.role == "wing":
            b.chain(name, p.role, p.side, p.chain, SHARES["wing"], parent)
        elif p.role == "tail":  # the tail of a standing figure (a lying one has it at the end of the spine)
            count = int(np.clip(round(p.length / (TAIL_BONE * size)), 2, 8))
            b.chain("tail", "tail", "", p.chain, [1] * count, spine_bones[0])
        elif p.role == "arm":
            other = pair_len.get({"L": "R", "R": "L"}.get(p.side, ""), 0.0)
            hand_reach = other if other and p.length > PROP_RATIO * other else 0.0
            if not hand_reach and p.length > 0.75 * body.height and len(arms) == 1:
                hand_reach = 0.45 * body.height
            if hand_reach:  # the arm up to where its pair ends, the rest is held in the hand
                arc = _arc(pts[p.chain])
                k = int(np.searchsorted(arc, hand_reach))
                arm_bones = b.chain(name, "arm", p.side, p.chain[: k + 1], SHARES["arm"], parent)
                b.chain("prop", "prop", p.side, p.chain[k:], [1], arm_bones[-1] if arm_bones else parent)
            else:
                b.chain(name, "arm", p.side, p.chain, SHARES["arm"], parent)
        else:
            count = int(np.clip(round(p.length / (EXTRA_BONE * size)), 1, 3))
            b.chain(name, "extra", p.side, p.chain, [1] * count, parent)

    _twigs(b, body)
    return b.bones


def _twigs(b: _Builder, body: Body) -> None:
    """Long side branches of a part (antlers on a head, a sword beside the hand) get chains of their own."""
    from .roles import _main_chain, _subtree

    tree, size = b.tree, body.size
    number = 1 + sum(1 for p in body.parts if p.role == "extra")
    for p in body.parts:
        chain_set = set(p.chain)
        key = next((k for k, ids in b.node_ids.items() if p.tip in ids), None)
        if key is None:
            continue
        arc = _arc(tree.points[p.chain])
        for c, s in zip(p.chain[1:], arc[1:]):
            if s < TWIG_FROM * arc[-1]:  # near the body a side branch is flesh of the body, not something held
                continue
            for j in tree.nbrs[c]:
                if j in chain_set or j not in p.nodes:
                    continue
                sub = _subtree(tree, j, chain_set)
                twig = _main_chain(tree, c, set(sub))
                if tree.length(twig) < TWIG * size:
                    continue
                side = p.side
                count = int(np.clip(round(tree.length(twig) / (EXTRA_BONE * size)), 1, 3))
                b.chain(f"extra{number}", "extra", side, twig, [1] * count, b.bone_at(key, c))
                number += 1


def as_dicts(bone_list: list[Bone]) -> list[dict]:
    out = []
    for bone in bone_list:
        d = asdict(bone)
        d["head"] = [round(float(x), 5) for x in d["head"]]
        d["tail"] = [round(float(x), 5) for x in d["tail"]]
        out.append(d)
    return out


def summary(body: Body) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in body.parts:
        out[p.role] = out.get(p.role, 0) + 1
    return out


__all__ = ["Bone", "bones", "as_dicts", "summary", "Part"]
