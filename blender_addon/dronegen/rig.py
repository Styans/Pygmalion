"""Skeletons in Blender: bones from the server -> an Armature; the figure and its drone points bound to it.

The mesh goes to the server turned and scaled as the scene shows it (world axes: Z is up, whatever the user
did to the object), its origin kept; the bones come back in the same frame, so the armature sits at the
mesh's origin without a turn and the mesh becomes its child without moving. Binding: Blender's automatic weights («bone heat»)
on a lighter copy of the mesh, carried over to the full mesh and to the Pointille drone points by their
nearest vertices; where heat finds no answer, weights by the distance to the bones. Both get an Armature
modifier. Skybrush pins every drone to a vertex group of the points, and such a pin follows the deformed
mesh, so the drones follow the bones (checked: Copy Location to a vertex group follows an Armature modifier
and Geometry Nodes after it).

Bones carry their role, side and chain (``dg_role``, ``dg_side``, ``dg_chain``, ``dg_segment``): motions and
effects find «the left wing» by these, never by names, which the user may change.
"""

from __future__ import annotations

import io
import json
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

RIG_KEY = "dronegen_rig"  # on the armature: JSON with the body's axes, parts and notes from the server
POINTS_KEY = "pointille_drones"  # Pointille marks its drone points with it
SOURCES_KEY = "pointille_sources"
PROXY_FACES = 30000  # bone heat on a lighter copy: seconds instead of minutes, the same weights to the eye
NEIGHBOURS = 4  # a vertex takes the weights of this many nearest vertices of the copy
MIN_WEIGHT = 0.02
WEIGHT_STEPS = 64  # weights are written in steps of 1/64: one call per bone and step instead of per vertex

ROLE_GROUPS = {  # bone collections, so that a part can be shown alone
    "root": "Корень", "spine": "Тело", "neck": "Голова и шея", "head": "Голова и шея", "tail": "Хвост",
    "leg_front": "Ноги", "leg_hind": "Ноги", "leg": "Ноги", "arm": "Руки", "prop": "Руки", "wing": "Крылья",
    "extra": "Прочее",
}
SIDE_COLORS = {"L": "THEME01", "R": "THEME04"}  # left red, right blue (Blender's bone color sets)
PART_NAMES = {"wing": "крыло", "leg_front": "передняя нога", "leg_hind": "задняя нога", "leg": "нога", "arm": "рука",
              "prop": "предмет в руке", "tail": "хвост", "neck": "шея", "head": "голова", "spine": "тело",
              "root": "тело", "extra": "часть"}
SIDE_WORDS = {"L": " (левое)", "R": " (правое)"}
PART_PREFIX = "DG· "  # on drone points: a vertex group per part of the body, for light by parts (Pointille)
PART_ATTRS = ("dg_part", "dg_along")  # which part a drone is on, and how far along it from its root (0..1)
PARTS_KEY = "dronegen_parts"  # on drone points: JSON list of the parts' groups, by dg_part


# -- finding things -------------------------------------------------------------------------
def is_points(obj) -> bool:
    return obj is not None and obj.type == "MESH" and bool(obj.get(POINTS_KEY))


def is_rig(obj) -> bool:
    return obj is not None and obj.type == "ARMATURE" and RIG_KEY in obj


def rig_of(mesh) -> bpy.types.Object | None:
    for m in mesh.modifiers:
        if m.type == "ARMATURE" and is_rig(m.object):
            return m.object
    return mesh.parent if is_rig(mesh.parent) else None


def figure_of(obj) -> bpy.types.Object | None:
    """The figure mesh behind whatever is selected: the mesh itself, its drone points, its rig."""
    if obj is None:
        return None
    if is_rig(obj):
        return next((c for c in obj.children if c.type == "MESH" and not is_points(c) and rig_of(c) is obj), None)
    if is_points(obj):
        if obj.parent is not None and obj.parent.type == "MESH" and not is_points(obj.parent):
            return obj.parent
        names = [n for n in str(obj.get(SOURCES_KEY, "")).split("\n") if n]
        return next((bpy.data.objects[n] for n in names if n in bpy.data.objects), None)
    return obj if obj.type == "MESH" else None


def drone_points(mesh) -> list:
    """Pointille points made on this figure: its children, or objects naming it as their source."""
    out = []
    for o in bpy.data.objects:
        if not is_points(o):
            continue
        sources = str(o.get(SOURCES_KEY, "")).split("\n")
        if o.parent is mesh or mesh.name in sources:
            out.append(o)
    return out


def info(rig) -> dict:
    try:
        return json.loads(rig.get(RIG_KEY) or "{}")
    except ValueError:
        return {}


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


PART_WORDS = {  # role -> (one, few, many)
    "leg": ("нога", "ноги", "ног"), "wing": ("крыло", "крыла", "крыльев"), "arm": ("рука", "руки", "рук"),
    "extra": ("доп. часть", "доп. части", "доп. частей"),
}


def describe(data: dict) -> str:
    """«4 ноги, 2 крыла, хвост» from the server's ``parts``, ``rigid`` and the bones."""
    if data.get("rigid"):
        return "предмет без частей: одна цепочка"
    parts = data.get("parts") or {}
    counts = {"leg": sum(n for k, n in parts.items() if k.startswith("leg")), "wing": parts.get("wing", 0),
              "arm": parts.get("arm", 0)}
    out = [f"{n} {_plural(n, *PART_WORDS[k])}" for k, n in counts.items() if n]
    if parts.get("tail") or any(b.get("role") == "tail" for b in data.get("bones", [])):
        out.append("хвост")
    if parts.get("extra"):
        out.append(f"{parts['extra']} {_plural(parts['extra'], *PART_WORDS['extra'])}")
    return ", ".join(out) or "только тело"


def bound(obj, rig) -> bool:
    return any(m.type == "ARMATURE" and m.object is rig for m in obj.modifiers)


# -- to the server --------------------------------------------------------------------------
def mesh_npz(mesh) -> bytes:
    """The figure as it is now (without modifiers) in world axes around its origin: what the server fits the bones
    into. The server takes Z for up, and a turned object's own Z need not be up."""
    me = mesh.data
    me.calc_loop_triangles()
    v = _coords(me) @ np.array(mesh.matrix_world.to_3x3(), np.float64).T
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    buf = io.BytesIO()
    np.savez_compressed(buf, vertices=v.astype(np.float32), faces=t.reshape(-1, 3))
    return buf.getvalue()


# -- the armature ---------------------------------------------------------------------------
def _override(**kw):
    """A context for operators, also from a timer (no window there)."""
    wm = bpy.context.window_manager
    window = bpy.context.window or (wm.windows[0] if wm is not None and wm.windows else None)
    if window is not None and "window" not in kw:
        kw["window"] = window
        area = next((a for a in window.screen.areas if a.type == "VIEW_3D"), None)
        if area is not None:
            kw["area"] = area
            kw["region"] = next((r for r in area.regions if r.type == "WINDOW"), None)
    return bpy.context.temp_override(**kw)


def _set_mode(obj, mode: str) -> None:
    view_layer = bpy.context.view_layer
    view_layer.objects.active = obj
    if obj.mode != mode:
        with _override(active_object=obj, object=obj):
            bpy.ops.object.mode_set(mode=mode)


def build(context, mesh, result: dict) -> bpy.types.Object:
    """A new armature for ``mesh`` from the server's answer; an old DroneGen rig of the mesh is replaced."""
    old = rig_of(mesh)
    if old is not None:
        remove(mesh, old)
    if context.mode != "OBJECT" and context.object is not None:
        _set_mode(context.object, "OBJECT")
    data = bpy.data.armatures.new(f"{mesh.name} Rig")
    data.display_type = "OCTAHEDRAL"
    rig = bpy.data.objects.new(f"{mesh.name} Rig", data)
    for coll in mesh.users_collection:
        coll.objects.link(rig)
    rig.matrix_world = Matrix.Translation(mesh.matrix_world.translation)
    rig.show_in_front = True
    keep = {k: result.get(k) for k in ("frame", "upright", "rigid", "ground", "height", "size", "parts", "notes",
                                      "plan", "method")}
    rig[RIG_KEY] = json.dumps(keep, ensure_ascii=False)

    _set_mode(rig, "EDIT")
    up, forward = Vector(result["frame"]["up"]), Vector(result["frame"]["forward"])
    made = {}
    for b in result["bones"]:
        eb = data.edit_bones.new(b["name"])
        eb.head, eb.tail = Vector(b["head"]), Vector(b["tail"])
        if (eb.tail - eb.head).length < 1e-6:
            eb.tail = eb.head + up * 1e-3
        eb.use_deform = bool(b.get("deform", True))
        # Z of a bone faces up (forward for upright bones): rolls are predictable, poses mirror cleanly
        direction = (eb.tail - eb.head).normalized()
        eb.align_roll(forward if abs(direction.dot(up)) > 0.7 else up)
        for key in ("role", "side", "chain", "segment"):
            eb[f"dg_{key}"] = b[key]
        made[b["name"]] = eb
    for b in result["bones"]:
        if b.get("parent") in made:
            eb, parent = made[b["name"]], made[b["parent"]]
            eb.parent = parent
            eb.use_connect = parent.get("dg_chain") == eb.get("dg_chain") and (parent.tail - eb.head).length < 1e-5
    _set_mode(rig, "OBJECT")
    _collections(rig, result["bones"])
    return rig


def _collections(rig, bones: list) -> None:
    data = rig.data
    if not hasattr(data, "collections"):  # Blender before 4.0
        return
    for b in bones:
        name = ROLE_GROUPS.get(b["role"], "Прочее")
        coll = data.collections.get(name) or data.collections.new(name)
        bone = data.bones[b["name"]]
        coll.assign(bone)
        if b["side"] in SIDE_COLORS and hasattr(bone, "color"):
            bone.color.palette = SIDE_COLORS[b["side"]]


def remove(mesh, rig) -> None:
    """Unbind the figure and its points from ``rig`` and delete the rig."""
    unbind(mesh, rig)
    world = mesh.matrix_world.copy()
    if mesh.parent is rig:
        mesh.parent = None
        mesh.matrix_world = world
    data = rig.data
    bpy.data.objects.remove(rig)
    if data.users == 0:
        bpy.data.armatures.remove(data)


def unbind(mesh, rig) -> None:
    names = {b.name for b in rig.data.bones}
    for obj in [mesh, *drone_points(mesh)]:
        for m in [m for m in obj.modifiers if m.type == "ARMATURE" and m.object is rig]:
            obj.modifiers.remove(m)
        for vg in [vg for vg in obj.vertex_groups if vg.name in names or vg.name.startswith(PART_PREFIX)]:
            obj.vertex_groups.remove(vg)
        for attr in PART_ATTRS:
            if attr in obj.data.attributes:
                obj.data.attributes.remove(obj.data.attributes[attr])
        if PARTS_KEY in obj:
            del obj[PARTS_KEY]


# -- binding --------------------------------------------------------------------------------
def bind(context, mesh, rig) -> dict:
    """Weights for the figure and its drone points, Armature modifiers, the figure parented to the rig."""
    t0 = time.time()
    if context.mode != "OBJECT" and context.object is not None:
        _set_mode(context.object, "OBJECT")
    unbind(mesh, rig)
    bones = [b for b in rig.data.bones if b.use_deform]
    names = [b.name for b in bones]
    proxy = _proxy(context, mesh)
    pose = rig.data.pose_position
    rig.data.pose_position = "REST"  # weights are worked out on the skeleton at rest, whatever pose it is in now
    try:
        heat = _heat(proxy, rig, names)
        proxy_co = _coords(proxy.data)
    finally:
        rig.data.pose_position = pose
        data = proxy.data
        bpy.data.objects.remove(proxy)
        bpy.data.meshes.remove(data)
    proxy_co = _apply(_to_rig(rig, mesh), proxy_co)  # everything below is in the armature's space
    segments = np.array([[b.head_local, b.tail_local] for b in bones], np.float64)
    unweighted = heat.sum(axis=1) <= 0
    if unweighted.any():  # heat found no answer here: the nearest bones by distance
        heat[unweighted] = _by_distance(proxy_co[unweighted], segments)

    targets = [mesh, *drone_points(mesh)]
    from mathutils.kdtree import KDTree

    tree = KDTree(len(proxy_co))
    for i, co in enumerate(proxy_co):
        tree.insert(co, i)
    tree.balance()
    counts = {}
    for obj in targets:
        co = _coords(obj.data)
        if not len(co):
            continue
        local = _apply(_to_rig(rig, obj), co)
        weights = _carry(tree, local, heat)
        _write(obj, names, weights)
        if obj is not mesh:
            parts(obj, bones, local, weights)
        _armature_modifier(obj, rig)
        counts[obj.name] = len(co)

    world = mesh.matrix_world.copy()
    mesh.parent = rig
    mesh.matrix_parent_inverse = rig.matrix_world.inverted()
    mesh.matrix_world = world
    return {"seconds": round(time.time() - t0, 1), "objects": counts, "fallback": int(unweighted.sum()),
            "proxy": len(proxy_co)}


def _to_rig(rig, obj) -> np.ndarray:
    """Object space -> the armature's space (constant while the object hangs on the rig)."""
    return np.array(rig.matrix_world.inverted() @ obj.matrix_world, np.float64)


def _apply(m: np.ndarray, co: np.ndarray) -> np.ndarray:
    return co @ m[:3, :3].T + m[:3, 3]


def _coords(me) -> np.ndarray:
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _proxy(context, mesh):
    """A lighter copy of the figure (no modifiers) in the same place, for bone heat."""
    tmp = bpy.data.objects.new("DG weights", mesh.data)
    context.scene.collection.objects.link(tmp)
    tmp.matrix_world = mesh.matrix_world
    faces = max(1, len(mesh.data.polygons))
    if faces > PROXY_FACES:
        dec = tmp.modifiers.new("decimate", "DECIMATE")
        dec.ratio = PROXY_FACES / faces
    depsgraph = context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(depsgraph))
    bpy.data.objects.remove(tmp)
    proxy = bpy.data.objects.new("DG weights", me)
    context.scene.collection.objects.link(proxy)
    proxy.matrix_world = mesh.matrix_world
    return proxy


def _heat(proxy, rig, names: list) -> np.ndarray:
    """Blender's automatic weights on the copy: (vertices, bones), zero rows where heat found nothing."""
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    proxy.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    with _override(active_object=rig, object=rig, selected_objects=[proxy, rig],
                   selected_editable_objects=[proxy, rig]):
        bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    index = {vg.index: names.index(vg.name) for vg in proxy.vertex_groups if vg.name in names}
    heat = np.zeros((len(proxy.data.vertices), len(names)))
    for v in proxy.data.vertices:
        for g in v.groups:
            j = index.get(g.group)
            if j is not None:
                heat[v.index, j] = g.weight
    return heat


def _by_distance(points: np.ndarray, segments: np.ndarray, keep: int = 2) -> np.ndarray:
    """Weights of the ``keep`` nearest bones by the distance to them (1 / d⁴, normalised)."""
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    t = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0, 1)
    d = np.linalg.norm(points[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1)
    w = 1.0 / np.maximum(d, 1e-6) ** 4
    if w.shape[1] > keep:
        cut = np.partition(w, -keep, axis=1)[:, -keep][:, None]
        w = np.where(w >= cut, w, 0.0)
    return w / w.sum(axis=1, keepdims=True)


def _carry(tree, co: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weights of the copy's ``NEIGHBOURS`` nearest vertices, closer ones counting more."""
    out = np.zeros((len(co), weights.shape[1]))
    for i, p in enumerate(co):
        near = tree.find_n(p, NEIGHBOURS)
        total = 0.0
        for _co, j, dist in near:
            w = 1.0 / max(dist, 1e-6)
            out[i] += w * weights[j]
            total += w
        if total:
            out[i] /= total
    out[out < MIN_WEIGHT] = 0.0
    sums = out.sum(axis=1, keepdims=True)
    return np.divide(out, sums, out=np.zeros_like(out), where=sums > 0)


def _write(obj, names: list, weights: np.ndarray) -> None:
    for j, name in enumerate(names):
        col = weights[:, j]
        idx = np.nonzero(col > 0)[0]
        if not len(idx):
            continue
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        steps = np.maximum(1, np.round(col[idx] * WEIGHT_STEPS)).astype(int)
        for level in np.unique(steps):
            vg.add(idx[steps == level].tolist(), float(level) / WEIGHT_STEPS, "REPLACE")


def part_title(role: str, side: str) -> str:
    return PART_NAMES.get(role, "часть") + SIDE_WORDS.get(side, "")


def parts(obj, bones: list, co: np.ndarray, weights: np.ndarray) -> list[str]:
    """For light by parts of the body: each drone point goes to the part (chain of bones) it mostly follows: a
    vertex group «DG· крыло (левое)» per part, and the attributes ``dg_part`` (its number in the object's
    ``dronegen_parts`` list) and ``dg_along`` (0 at the part's root, 1 at its tip: a wave of light along a wing).
    ``co`` are the points in the armature's space, ``weights`` (points, bones) their bone weights."""
    chain_of = [b.get("dg_chain") or b.name for b in bones]
    chains = list(dict.fromkeys(chain_of))
    length = {c: sum(1 for x in chain_of if x == c) for c in chains}
    first = {c: min(b.get("dg_segment", 1) for b, x in zip(bones, chain_of) if x == c) for c in chains}
    titles, seen = {}, {}
    for c in chains:
        bone = bones[chain_of.index(c)]
        t = part_title(bone.get("dg_role", "extra"), bone.get("dg_side", ""))
        seen[t] = seen.get(t, 0) + 1
        titles[c] = t if seen[t] == 1 else f"{t} {seen[t]}"
    n = len(co)
    main = np.argmax(weights, axis=1) if weights.shape[1] else np.zeros(n, int)
    part = np.array([chains.index(chain_of[j]) for j in main], np.int32) if n else np.zeros(0, np.int32)
    along = np.zeros(n, np.float32)
    for i, j in enumerate(main):
        b, c = bones[j], chain_of[j]
        head, tail = np.array(b.head_local), np.array(b.tail_local)
        ab = tail - head
        t = float(np.clip((co[i] - head) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
        along[i] = (b.get("dg_segment", 1) - first[c] + t) / max(length[c], 1)
    used = [c for k, c in enumerate(chains) if (part == k).any()]
    for k, c in enumerate(chains):
        idx = np.nonzero(part == k)[0]
        if len(idx):
            vg = obj.vertex_groups.get(PART_PREFIX + titles[c]) or obj.vertex_groups.new(name=PART_PREFIX + titles[c])
            vg.add(idx.tolist(), 1.0, "REPLACE")
    for name, kind, values in (("dg_part", "INT", part), ("dg_along", "FLOAT", along)):
        if name in obj.data.attributes:
            obj.data.attributes.remove(obj.data.attributes[name])
        obj.data.attributes.new(name, kind, "POINT").data.foreach_set("value", values)
    obj[PARTS_KEY] = json.dumps([PART_PREFIX + titles[c] for c in chains], ensure_ascii=False)
    return [PART_PREFIX + titles[c] for c in used]


def _armature_modifier(obj, rig) -> None:
    mod = obj.modifiers.new("DroneGen Rig", "ARMATURE")
    mod.object = rig
    mod.use_vertex_groups = True
    mod.use_bone_envelopes = False
    if obj.modifiers[0] is not mod:  # first: effects on the points (Geometry Nodes) come after the bones
        with _override(object=obj, active_object=obj):
            bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
