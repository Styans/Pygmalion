"""Effects of animation layers: effects of shape on the figure's drone points, effects with drones of their own.

Effects of shape (``SHAPE``) are Geometry Nodes modifiers on every drone point object of the figure, after its
Armature modifier: the points move on top of the skeleton's motion. Their ``Factor`` (the layer's strength and
fades) is keyed in a strip of the points object, like a flight's.

Effects with drones of their own (``EFFECT``) are point objects of their own: drones for Pointille and Skybrush
like any other (``pointille_drones``, the «Pointille» vertex group), so their colours and the formations work as
usual. The server lays out their places and waiting places in the reserve behind the figure (``/effects/build``);
the places are point attributes, the times are inputs of the ``nodes.effect_group`` modifier. The light of an
effect is left to Pointille: ``dg_on`` and ``dg_off`` (seconds) tell when each drone should shine (for a trail:
when the tip passed it).

Where an effect is: above the figure, at the tip of a part of its body when the layer begins (both fly with the
figure), or fixed in the sky where the figure was. A trail stays in the sky where the tip passed.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import nodes, rig as rig_mod, safety

PREFIX = "DG· "
EFFECT_KEY = "dronegen_effect_of"  # on an effect's points object: the name of its figure's rig


def _frame(rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = rig_mod.info(rig).get("frame") or {}
    return (np.asarray(f.get("forward", (0, -1, 0)), float), np.asarray(f.get("left", (1, 0, 0)), float),
            np.asarray(f.get("up", (0, 0, 1)), float))


def figure(rig):
    return rig_mod.figure_of(rig)


def spacing(scene) -> float:
    return float(safety.limits(scene)["distance"])


# the least size of an effect, in safety distances, for a shape one can tell (a star of ten drones, a ring of five...)
MIN_SIZE = {"star": 4.0, "ring": 2.0, "heart": 3.0, "ripple": 7.0, "sparkles": 2.0, "splash": 0.0, "trail": 0.0,
            "model": 2.0}


def min_size(effect: str, scene) -> float:
    return MIN_SIZE.get(effect, 0.0) * spacing(scene)


def effects_of(rig) -> list:
    return [o for o in bpy.data.objects if o.get(EFFECT_KEY) == rig.name]


def budget(rig, scene) -> dict:
    """Drones of the figure, of its effects, and the fleet in Skybrush (0 when there is none)."""
    mesh = figure(rig)
    own = sum(len(p.data.vertices) for p in rig_mod.drone_points(mesh)) if mesh is not None else 0
    extra = sum(len(o.data.vertices) for o in effects_of(rig))
    coll = None
    sk = getattr(scene, "skybrush", None)
    if sk is not None:
        coll = sk.settings.drone_collection or bpy.data.collections.get("Drones")
    return {"figure": own, "effects": extra, "fleet": len(coll.objects) if coll is not None else 0}


def clear(rig) -> None:
    """Effects an earlier «Применить» made: the shape modifiers and their tracks on the points, the effects' objects."""
    mesh = figure(rig)
    for points in rig_mod.drone_points(mesh) if mesh is not None else []:
        for m in [m for m in points.modifiers if m.name.startswith(PREFIX)]:
            points.modifiers.remove(m)
        ad = points.animation_data
        if ad is not None:
            for track in [t for t in ad.nla_tracks if t.name.startswith(PREFIX)]:
                ad.nla_tracks.remove(track)
    for obj in effects_of(rig):
        data = obj.data
        bpy.data.objects.remove(obj)
        if data is not None and data.users == 0:
            bpy.data.meshes.remove(data)


# -- effects of shape ------------------------------------------------------------------------
def _local(obj, vector) -> tuple:
    """A world direction in the object's own axes (unit length)."""
    v = obj.matrix_world.inverted().to_3x3() @ Vector(vector)
    return tuple(v.normalized())


def shape_layer(context, rig, i: int, layer, rate: float, keys) -> int:
    """The layer's modifier on every drone point object of the figure; ``keys(obj, path, points)`` keys a value."""
    mesh = figure(rig)
    forward, _left, up = _frame(rig)
    size = float(rig_mod.info(rig).get("size") or max(rig.dimensions))
    made = 0
    for points in rig_mod.drone_points(mesh) if mesh is not None else []:
        scale = max(points.matrix_world.to_scale()) or 1.0
        mod = points.modifiers.new(f"{PREFIX}{i + 1} {layer.name}", "NODES")
        mod.node_group = nodes.shape_group(layer.shape_fx)
        kind = layer.shape_fx
        if kind == "WAVE":
            nodes.set_input(mod, "Amplitude", layer.fx_size / scale)
            nodes.set_input(mod, "Period", layer.period)
            nodes.set_input(mod, "Wavelength", size / scale)
            nodes.set_input(mod, "Direction", _local(points, up))
            nodes.set_input(mod, "Travel", _local(points, forward))
        elif kind == "PULSE":
            co = np.empty(len(points.data.vertices) * 3)
            points.data.vertices.foreach_get("co", co)
            nodes.set_input(mod, "Amplitude", min(layer.fx_size, 1.0) * 0.2)
            nodes.set_input(mod, "Period", layer.period)
            nodes.set_input(mod, "Center", tuple(co.reshape(-1, 3).mean(axis=0)) if len(co) else (0.0, 0.0, 0.0))
        elif kind == "JITTER":
            nodes.set_input(mod, "Amplitude", layer.fx_size / scale)
            nodes.set_input(mod, "Speed", 1.0 / max(layer.period, 0.1))
            nodes.set_input(mod, "Scale", 3.0 * scale / max(size, 1e-3))
        else:  # SCATTER
            nodes.set_input(mod, "Radius", layer.fx_size / scale)
            nodes.set_input(mod, "Seed", i)
        path = nodes.set_input(mod, "Factor", 0.0)
        keys(points, path, layer)
        made += 1
    return made


# -- effects with drones of their own ----------------------------------------------------------
def _chain_tip(rig, chain: str):
    bones = [b for b in rig.pose.bones if b.bone.get("dg_chain") == chain or b.name == chain]
    return max(bones, key=lambda b: b.bone.get("dg_segment", 1)) if bones else None


def _scale(world: np.ndarray) -> float:
    """Metres in one unit of the rig's space (a figure is often scaled up to its size in the sky)."""
    return float(np.linalg.norm(world[:3, :3], axis=0).mean()) or 1.0


def place(context, rig, layer) -> tuple[np.ndarray, np.ndarray]:
    """Where the effect is in the rig's space, and the rig's world matrix when the layer begins."""
    info = rig_mod.info(rig)
    up = _frame(rig)[2]
    keep = context.scene.frame_current
    context.scene.frame_set(layer.start)
    try:
        world = np.array(rig.matrix_world)
        s = _scale(world)  # the layer's sizes are metres
        top = float(info.get("ground", 0.0)) + float(info.get("height", 0.0))
        at = up * (top + (layer.size / 2 + layer.offset) / s)
        tip = _chain_tip(rig, layer.bone) if layer.anchor == "BONE" and layer.bone else None
        if tip is not None:
            at = np.array(tip.tail) + up * layer.offset / s
    finally:
        context.scene.frame_set(keep)
    return at, world


def lane(rig, layer) -> int:
    """Which of the figure's effects this is: each waits in a reserve of its own, one behind another."""
    layers = [x for x in rig.dronegen_layers if x.kind == "EFFECT"]
    return next((k for k, x in enumerate(layers) if x == layer), 0)


def behind(rig, at: np.ndarray, gap: float, lane_no: int = 0) -> np.ndarray:
    """From the effect to its reserve (the rig's space, ``gap`` too): straight behind the figure, out of the audience's
    sight, at the effect's own height, so the drones fly across, not up (drones climb much slower than they fly).
    The reserves of different effects stand two safety distances apart."""
    forward = _frame(rig)[0]
    size = float(rig_mod.info(rig).get("size") or 1.0)
    return forward * (-(size / 2 + (3 + 2 * lane_no) * gap) - float(at @ forward))


def _request(kind: str, layer, rig, gap: float, anchor, reserve_rel, follows: bool, turn: np.ndarray,
             scale: float) -> dict:
    """``turn`` takes the rig's directions to the effect object's axes; everything the server gets is in metres."""
    forward, left, up = (turn @ v for v in _frame(rig))
    return {"kind": kind, "size": layer.size, "spacing": gap, "count": layer.count,
            "frame": {"forward": forward.tolist(), "left": left.tolist(), "up": up.tolist()},
            "reserve_at": np.asarray(reserve_rel).tolist(), "params": {"seed": 0},
            "_anchor": np.asarray(anchor).tolist(), "_follows": follows, "_scale": scale}


def _world_request(kind: str, layer, rig, gap: float, at: np.ndarray, world: np.ndarray) -> dict:
    """An effect fixed in the sky where ``at`` (the rig's space) was when the layer began."""
    s = _scale(world)
    rel = world[:3, :3] @ behind(rig, at, gap / s, lane(rig, layer))
    return _request(kind, layer, rig, gap, (world @ np.append(at, 1.0))[:3], rel, False, world[:3, :3] / s, s)


def request(context, rig, layer) -> dict | None:
    """What to ask the server for an effect layer (made on the main thread; a trail waits for the animation)."""
    if layer.effect == "trail":
        return None
    gap = spacing(context.scene)
    at, world = place(context, rig, layer)
    if layer.anchor == "WORLD":
        req = _world_request(layer.effect, layer, rig, gap, at, world)
    else:  # flies with the figure: the object's axes are the rig's, its units metres
        s = _scale(world)
        rel = behind(rig, at, gap / s, lane(rig, layer)) * s
        req = _request(layer.effect, layer, rig, gap, at * s, rel, True, np.eye(3), s)
    if layer.effect == "splash":  # slow enough for drones: they climb at no more than 80 % of the limit
        lim = safety.limits(context.scene)
        lift = 1.2
        req["params"].update(speed=0.8 * lim["up"] * np.sqrt(1 + lift * lift) / lift, lift=lift)
    if layer.effect == "model":
        if layer.model is None:
            raise RuntimeError(f"Слой «{layer.name}»: выберите модель")
        req["kind"], req["points"] = "points", sample_model(context, layer.model, layer.size).tolist()
    return req


def sample_model(context, obj, size: float, n: int = 6000) -> np.ndarray:
    """Points on a model's surface (area-weighted), centred and scaled to ``size`` across."""
    ev = obj.evaluated_get(context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
    finally:
        ev.to_mesh_clear()
    co, tri = co.reshape(-1, 3), tri.reshape(-1, 3)
    if not len(tri):
        raise RuntimeError(f"У «{obj.name}» нет граней")
    m = np.array(obj.matrix_world.to_3x3())
    co = co @ m.T
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(tri), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    return pts * (size / max(np.ptp(pts, axis=0).max(), 1e-9))


def trail_path(context, rig, layer, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the tip of the layer's part goes (world) during the layer, frame by frame, and when (seconds)."""
    tip = _chain_tip(rig, layer.bone)
    if tip is None:
        raise RuntimeError(f"Слой «{layer.name}»: нет части «{layer.bone}» у скелета")
    scene, keep = context.scene, context.scene.frame_current
    pts, times = [], []
    try:
        for f in range(int(layer.start), int(max(layer.end, layer.start + 1)) + 1):
            scene.frame_set(f)
            pts.append(tuple(rig.matrix_world @ tip.tail))
            times.append(f / rate)
    finally:
        scene.frame_set(keep)
    return np.array(pts), np.array(times)


def trail_request(context, rig, layer, rate: float) -> dict:
    gap = spacing(context.scene)
    path, times = trail_path(context, rig, layer, rate)
    anchor = path[0]
    keep = context.scene.frame_current
    context.scene.frame_set(layer.start)
    world = np.array(rig.matrix_world)
    context.scene.frame_set(keep)
    at = (np.linalg.inv(world) @ np.append(anchor, 1.0))[:3]  # the trail's start in the rig's space
    # the path densely, so the server can keep drones at the safety distance along it
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    at_arc = np.linspace(0.0, arc[-1], max(2, int(arc[-1] / (gap / 8)) + 1))
    dense = np.stack([np.interp(at_arc, arc, path[:, k]) for k in range(3)], 1)
    req = _world_request("points", layer, rig, gap, at, world)
    req.update(size=float(max(np.ptp(path, axis=0).max(), gap)), points=(dense - anchor).tolist(),
               _path=(path - anchor).tolist(), _times=times.tolist())
    return req


def travel_time(scene, shape: np.ndarray, waiting: np.ndarray, up: np.ndarray) -> float:
    """Seconds the flights between the reserve and the places need within the drones' limits, 10 % to spare. The
    flight eases in and out (smoothstep over ``d`` in ``T``): its top speed is 1.5 d/T, its top acceleration 6 d/T²."""
    if not len(shape):
        return 0.0
    lim = safety.limits(scene)
    d = shape - waiting
    rise = d @ up
    across = np.linalg.norm(d - rise[:, None] * up, axis=1)
    need = 1.5 * max(float(across.max()) / lim["xy"], float(np.maximum(rise, 0).max()) / lim["up"],
                     float(np.maximum(-rise, 0).max()) / lim["down"])
    if lim["acceleration"] > 0:
        need = max(need, float(np.sqrt(6.0 * np.linalg.norm(d, axis=1).max() / lim["acceleration"])))
    return 1.1 * need


def effect_object(context, rig, i: int, layer, req: dict, geo: dict, rate: float, notes: list):
    """The effect's own drones, waiting in the reserve before and after; what the user should know goes to ``notes``."""
    shape = np.asarray(geo["shape"], np.float64).reshape(-1, 3)
    waiting = np.asarray(geo["reserve"], np.float64).reshape(-1, 3)
    n = len(shape)
    name = f"{PREFIX}{rig.name} · {i + 1} {layer.name}"
    me = bpy.data.meshes.new(name)
    me.from_pydata(waiting.tolist(), [], [])
    start, end = layer.start / rate, max(layer.end, layer.start + 1) / rate
    on = np.full(n, start)
    if req.get("_times"):  # a trail: each drone shines when the tip passes it
        path, times = np.asarray(req["_path"]), np.asarray(req["_times"])
        near = np.argmin(((shape[:, None, :] - path[None, :, :]) ** 2).sum(-1), axis=1)
        on = times[near]
        end += layer.fade_out  # it lingers a little after the swing
    velocity = np.asarray(geo.get("velocity") or np.zeros((n, 3)), np.float64)
    for attr, values, kind in (("dg_reserve", waiting, "FLOAT_VECTOR"), ("dg_shape", shape, "FLOAT_VECTOR"),
                               ("dg_velocity", velocity, "FLOAT_VECTOR"), ("dg_on", on, "FLOAT"),
                               ("dg_off", np.full(n, end), "FLOAT")):
        a = me.attributes.new(attr, kind, "POINT")
        a.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "value", np.asarray(values, np.float32).ravel())
    obj = bpy.data.objects.new(name, me)
    for coll in rig.users_collection:
        coll.objects.link(obj)
    anchor = np.asarray(req["_anchor"], np.float64)
    if req["_follows"]:  # flies with the figure, in metres however the rig is scaled
        obj.parent = rig
        s = 1.0 / float(req.get("_scale", 1.0))
        obj.matrix_parent_inverse = Matrix.Diagonal((s, s, s, 1.0))
    obj.location = tuple(anchor)
    obj[EFFECT_KEY] = rig.name
    obj["dronegen_effect_name"] = layer.name
    obj["pointille_drones"] = 1
    obj.vertex_groups.new(name="Pointille").add(list(range(n)), 1.0, "REPLACE")
    sb = getattr(obj, "skybrush", None)  # a Skybrush formation as it is: its drones are the group's vertices
    if sb is not None:
        sb.formation_vertex_group = "Pointille"
    mod = obj.modifiers.new(f"{PREFIX}эффект", "NODES")
    mod.node_group = nodes.effect_group()
    first = float(on.min()) if len(on) else start
    span = max(end - first, 1e-3)
    up = np.asarray(req["frame"]["up"], np.float64)
    expand = float(geo.get("params", {}).get("expand", 0.0))
    gravity = 2.0 * float(np.max(velocity @ up)) / span if geo.get("velocity") and n else 0.0  # back down by End
    last = shape * (1.0 + expand) + velocity * span - 0.5 * gravity * span * span * up  # where End finds them
    nodes.set_input(mod, "Start", first)
    nodes.set_input(mod, "End", float(end))
    need = max(travel_time(context.scene, shape, waiting, up), travel_time(context.scene, last, waiting, up))
    travel = max(float(layer.travel), need)
    if travel > layer.travel + 0.05:
        notes.append(f"«{layer.name}»: перелёт из резерва {travel:.1f} с вместо {layer.travel:.1f} — быстрее дроны "
                     f"не успеют в пределах скорости Skybrush")
    if first - travel < context.scene.frame_start / rate - 1e-6:
        notes.append(f"«{layer.name}»: дронам эффекта нужно вылететь из резерва раньше начала сцены — начните слой "
                     f"позже (не раньше кадра {int(np.ceil((context.scene.frame_start / rate + travel) * rate))})")
    lim = safety.limits(context.scene)
    spread = float(np.linalg.norm(last - shape, axis=1).max()) if n else 0.0
    if lim["acceleration"] > 0 and 6.0 * spread / (span * span) > lim["acceleration"]:  # smoothstep, as the flights
        notes.append(f"«{layer.name}»: за {span:.1f} с дроны разлетаются слишком резко — сделайте слой длиннее "
                     f"(от {np.sqrt(6.0 * spread / lim['acceleration']):.1f} с) или меньше")
    nodes.set_input(mod, "Travel", travel)
    nodes.set_input(mod, "Expand", expand)
    nodes.set_input(mod, "Gravity", gravity)
    return obj
