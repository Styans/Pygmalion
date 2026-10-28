"""Animation layers of a figure -> Blender's NLA: each layer is a track of its own, mixed with the others.

The layers live on the skeleton (``Object.dronegen_layers``). Motions come from the server's catalog
(``/motions``) and are baked for this skeleton there (``/motion/bake``): local quaternions of the bones a motion
turns, at times 0..1 of its period. Each motion layer becomes an action and one NLA strip:

* strips blend with Combine, so a wing beat, a gallop and breathing on different bones (or the same) add up;
* the layer's strength and its fades are the strip's animated influence: fading into a pose and out of it is a
  smooth transition, a pose layer after another is a sequence;
* a cycle's curves have a Cycles modifier: the strip lasts as long as the layer, the loop has no seam.

A flight layer is a Follow Path constraint on the skeleton (the mesh and its drone points hang on it), with
its position along the path and its fades keyed in a strip of its own that holds outside its span, so the
figure stays home before and after. Everything made here is named «DG· …»: applying again replaces only that,
the user's own tracks, actions and constraints stay.
"""

from __future__ import annotations

import json

import bpy
import numpy as np

from . import fx
from . import rig as rig_mod

PREFIX = "DG· "
CATALOG: dict = {}  # motion key -> its description from the server (filled by ``fetch``)
SIDES = {"R": -1, "L": 1, "BOTH": 0}


def fetch(client) -> dict:
    """The server's motion catalog (kept for the panel's menus)."""
    data = client.motions()
    CATALOG.clear()
    CATALOG.update(data)
    return CATALOG


def roles(rig) -> set[str]:
    return {b.get("dg_role", "extra") for b in rig.data.bones}


def fitting(rig) -> list[tuple[str, dict]]:
    """Motions of the catalog this skeleton can do."""
    have = roles(rig)
    return [(k, m) for k, m in CATALOG.items() if any(r in have or (r == "head" and "neck" in have) for r in m["roles"])]


def title(layer) -> str:
    if layer.kind == "PATH":
        return {"CIRCLE": "Полёт по кругу", "LINE": "Полёт прямо", "CURVE": "Полёт по кривой"}[layer.path_shape]
    m = CATALOG.get(layer.motion)
    return m["title"] if m else (layer.motion or "движение не выбрано")


def fps(scene) -> float:
    return scene.render.fps / scene.render.fps_base


def skeleton(rig) -> dict:
    """The skeleton as the server's ``motions.skeleton`` reads it: roles, rest matrices, the body's axes."""
    data = rig_mod.info(rig)
    bones = []
    for b in rig.data.bones:
        bones.append({"name": b.name, "role": b.get("dg_role", "extra"), "side": b.get("dg_side", ""),
                      "chain": b.get("dg_chain", b.name), "segment": int(b.get("dg_segment", 1)),
                      "head": list(b.head_local), "tail": list(b.tail_local),
                      "parent": b.parent.name if b.parent else None,
                      "matrix": [list(row) for row in b.matrix_local.to_3x3()]})
    return {"frame": data.get("frame") or {"forward": [0, -1, 0], "left": [1, 0, 0], "up": [0, 0, 1]},
            "size": data.get("size") or max(rig.dimensions), "bones": bones}


def params(layer) -> dict:
    """The layer's settings as the motion's parameters: «Размах» scales its first (main) parameter."""
    m = CATALOG.get(layer.motion) or {}
    out = {}
    names = list((m.get("params") or {}).keys())
    if names and names[0] != "side":
        p = m["params"][names[0]]
        out[names[0]] = float(min(max(p["default"] * layer.amplitude, p["min"]), p["max"]))
    if "side" in names:
        out["side"] = SIDES[layer.side]
    return out


def requests(rig) -> list[tuple[int, str, dict]]:
    """(layer index, motion, parameters) of every motion layer to bake."""
    out = []
    for i, layer in enumerate(rig.dronegen_layers):
        if layer.enabled and layer.kind == "MOTION" and layer.motion:
            out.append((i, layer.motion, params(layer)))
    return out


def effect_requests(context, rig) -> dict:
    """What to ask the server for every effect layer, by layer index (on the main thread: it reads the scene).
    A trail is asked for later: it follows the animation, which is not built yet."""
    out = {}
    for i, layer in enumerate(rig.dronegen_layers):
        if layer.enabled and layer.kind == "EFFECT":
            req = fx.request(context, rig, layer)
            if req is not None:
                out[str(i)] = req
    return out


def _public(req: dict) -> dict:
    return {k: v for k, v in req.items() if not k.startswith("_")}


def bake_all(client, skeleton_data: dict, todo: list, effects: dict | None = None) -> dict:
    """In a thread: the keys of every motion layer and the places of every effect, by layer index."""
    return {"motions": {str(i): client.bake(skeleton_data, motion, p) for i, motion, p in todo},
            "effects": {i: client.effect(_public(req)) for i, req in (effects or {}).items()}}


def trails(context, rig) -> dict:
    """Requests for trail effects: the tip's path is sampled from the built animation."""
    rate = fps(context.scene)
    return {str(i): fx.trail_request(context, rig, layer, rate) for i, layer in enumerate(rig.dronegen_layers)
            if layer.enabled and layer.kind == "EFFECT" and layer.effect == "trail"}


def place_effects(context, rig, requests: dict, geometry: dict, notes: list | None = None) -> int:
    rate = fps(context.scene)
    notes = [] if notes is None else notes
    made = 0
    for i, geo in geometry.items():
        layer = rig.dronegen_layers[int(i)]
        notes.extend(geo.get("notes") or [])
        fx.effect_object(context, rig, int(i), layer, requests[i], geo, rate, notes)
        made += 1
    return made


# -- building the NLA -------------------------------------------------------------------------
def clear(rig) -> None:
    """Remove everything an earlier «Применить» made: tracks, actions, constraints and paths named «DG· …»."""
    ad = rig.animation_data
    if ad is not None:
        if ad.action is not None and ad.action.name.startswith(PREFIX):
            ad.action = None
        for track in [t for t in ad.nla_tracks if t.name.startswith(PREFIX)]:
            ad.nla_tracks.remove(track)
    for c in [c for c in rig.constraints if c.name.startswith(PREFIX)]:
        rig.constraints.remove(c)
    for act in [a for a in bpy.data.actions if a.name.startswith(PREFIX + rig.name)]:
        if act.users == 0:
            bpy.data.actions.remove(act)
    for obj in [o for o in bpy.data.objects if o.get("dronegen_path_of") == rig.name]:
        data = obj.data
        bpy.data.objects.remove(obj)
        if data is not None and data.users == 0:
            bpy.data.curves.remove(data)
    fx.clear(rig)


def build(context, rig, result: dict, requests: dict | None = None) -> dict:
    """Tracks, effects of shape and effects with drones for every enabled layer, from what the server answered
    (``bake_all``) and the effect requests made before it (``effect_requests``). Returns what was made."""
    clear(rig)
    baked = result.get("motions", {})
    rate = fps(context.scene)
    ad = rig.animation_data or rig.animation_data_create()
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    made = {"motions": 0, "paths": 0, "shapes": 0, "effects": 0}
    for i, layer in enumerate(rig.dronegen_layers):
        if not layer.enabled:
            continue
        if layer.kind == "MOTION" and str(i) in baked:
            _motion_track(rig, ad, i, layer, baked[str(i)], rate)
            made["motions"] += 1
        elif layer.kind == "PATH":
            if _path_track(context, rig, ad, i, layer, rate):
                made["paths"] += 1
        elif layer.kind == "SHAPE":
            def keys(obj, path, layer=layer, i=i):
                _held(obj, f"{i + 1} {layer.name}", path, _fades(float(layer.start), float(max(layer.end, layer.start + 1)),
                                                                     layer.fade_in * rate, layer.fade_out * rate,
                                                                     layer.strength), name=f"{obj.name} · {i + 1}")
            if fx.shape_layer(context, rig, i, layer, rate, keys):
                made["shapes"] += 1
    made["notes"] = []
    made["effects"] = place_effects(context, rig, requests or {}, result.get("effects", {}), made["notes"])
    rig["dronegen_animation"] = json.dumps({"layers": len(rig.dronegen_layers), **made})
    return made


def _action(rig, i: int, layer) -> bpy.types.Action:
    name = f"{PREFIX}{rig.name} · {i + 1} {layer.name}"
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    return bpy.data.actions.new(name)


def _fcurve(action, rig, path: str, index: int, group: str):
    if not hasattr(action, "slots"):  # Blender before 4.4: curves right in the action
        return action.fcurves.find(path, index=index) or action.fcurves.new(path, index=index, action_group=group)
    from bpy_extras import anim_utils  # slotted actions: curves live in a channel bag of the slot for this object

    slot = action.slots[0] if len(action.slots) else action.slots.new(id_type="OBJECT", name=rig.name)
    bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    fc = bag.fcurves.find(path, index=index) or bag.fcurves.new(path, index=index)
    fc.group = bag.groups.get(group) or bag.groups.new(group)
    return fc


def _keys(fc, frames: list, values: list, cyclic: bool, linear: bool = False) -> None:
    fc.keyframe_points.add(len(frames))
    co = np.empty(2 * len(frames), np.float32)
    co[0::2], co[1::2] = frames, values
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR" if linear else "BEZIER"
        kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
    if cyclic:  # also makes the handles at the seam see the other end of the loop
        fc.modifiers.new("CYCLES")
    fc.update()


def _strip(ad, rig, name: str, action, start: float, length: float, blend: str, extrapolation: str):
    track = ad.nla_tracks.new()
    track.name = PREFIX + name
    strip = track.strips.new(name, int(round(start)), action)
    if hasattr(strip, "action_slot") and strip.action_slot is None and len(action.slots):
        strip.action_slot = action.slots[0]
    strip.use_sync_length = False
    strip.action_frame_start = 0.0
    strip.action_frame_end = max(1.0, float(length))
    strip.blend_type = blend
    strip.extrapolation = extrapolation
    return strip


def _fades(start: float, end: float, fade_in: float, fade_out: float, top: float) -> list[tuple[float, float]]:
    fade_in = min(fade_in, 0.5 * (end - start))
    fade_out = min(fade_out, 0.5 * (end - start))
    return [(start, 0.0 if fade_in > 0 else top), (start + fade_in, top), (end - fade_out, top),
            (end, 0.0 if fade_out > 0 else top)]


def _motion_track(rig, ad, i: int, layer, data: dict, rate: float) -> None:
    kind = data["kind"]
    period = layer.period * rate
    action = _action(rig, i, layer)
    frames = [t * period for t in data["times"]]
    cyclic = kind == "cycle"
    for bone, keys in data["bones"].items():
        if bone not in rig.pose.bones:
            continue
        path = f'pose.bones["{bone}"].rotation_quaternion'
        for c in range(4):
            _keys(_fcurve(action, rig, path, c, bone), frames, [k[c] for k in keys], cyclic)
    root = data.get("root")
    if root and root["bone"] in rig.pose.bones:
        path = f'pose.bones["{root["bone"]}"].location'
        for c in range(3):
            _keys(_fcurve(action, rig, path, c, root["bone"]), frames, [v[c] for v in root["location"]], cyclic)
    start = float(layer.start)
    end = start + period if kind == "once" else float(max(layer.end, layer.start + 1))
    strip = _strip(ad, rig, f"{i + 1} {layer.name}", action, start, end - start, "COMBINE", "NOTHING")
    strip.use_animated_influence = True
    for frame, value in _fades(start, end, layer.fade_in * rate, layer.fade_out * rate, layer.strength):
        strip.influence = value
        strip.keyframe_insert("influence", frame=frame)


_AXES = {(0, 1): "FORWARD_X", (1, 1): "FORWARD_Y", (2, 1): "FORWARD_Z", (0, -1): "TRACK_NEGATIVE_X",
         (1, -1): "TRACK_NEGATIVE_Y", (2, -1): "TRACK_NEGATIVE_Z"}


def _frame(rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = rig_mod.info(rig).get("frame") or {}
    return (np.asarray(f.get("forward", (0, -1, 0)), float), np.asarray(f.get("left", (1, 0, 0)), float),
            np.asarray(f.get("up", (0, 0, 1)), float))


def _path_curve(rig, i: int, layer) -> bpy.types.Object | None:
    """The layer's path: the chosen curve, or a circle / a straight line starting where the figure is and heading
    where it faces. The Follow Path constraint adds the curve's point to the figure's own place."""
    if layer.path_shape == "CURVE":
        return layer.curve
    forward, left, _up = _frame(rig)
    name = f"{PREFIX}{rig.name} · путь {i + 1}"
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    if layer.path_shape == "CIRCLE":
        r = layer.radius
        spline = cu.splines.new("BEZIER")
        spline.bezier_points.add(3)
        for k, p in enumerate(spline.bezier_points):
            # Follow Path measures a closed curve from its last point (checked in Blender 5.2): the figure's own
            # place goes last, so the flight starts where it stands, heading forward, turning left
            a = np.pi / 2 * (k + 1)
            p.co = tuple(r * left + r * (-np.cos(a) * left + np.sin(a) * forward))
            p.handle_left_type = p.handle_right_type = "AUTO"
        spline.use_cyclic_u = True
    else:
        spline = cu.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
        spline.points[1].co = (*(layer.radius * forward), 1.0)
    obj = bpy.data.objects.new(name, cu)
    for coll in rig.users_collection:
        coll.objects.link(obj)
    obj["dronegen_path_of"] = rig.name
    obj.hide_render = True
    return obj


def _held(obj, track: str, path: str, points: list[tuple[float, float]], name: str) -> None:
    """Key one value of ``obj`` at scene frames, in a strip of its own that holds before and after (so the value
    stays at its first key before the layer, at its last after)."""
    action = bpy.data.actions.get(PREFIX + name)
    if action is not None:
        bpy.data.actions.remove(action)
    action = bpy.data.actions.new(PREFIX + name)
    _keys(_fcurve(action, obj, path, 0, "DroneGen"), [f for f, _ in points], [v for _, v in points], cyclic=False)
    ad = obj.animation_data or obj.animation_data_create()
    _strip(ad, obj, track, action, 0.0, points[-1][0] + 1.0, "REPLACE", "HOLD")


def _path_track(context, rig, ad, i: int, layer, rate: float) -> bool:
    curve = _path_curve(rig, i, layer)
    if curve is None:
        return False
    forward, _left, _up = _frame(rig)
    axis = int(np.argmax(np.abs(forward)))
    c = rig.constraints.new("FOLLOW_PATH")
    c.name = f"{PREFIX}{i + 1} {layer.name}"
    c.target = curve
    c.use_curve_follow = True
    c.use_fixed_location = True
    c.forward_axis = _AXES[(axis, 1 if forward[axis] > 0 else -1)]
    c.up_axis = "UP_Z"
    c.influence = 0.0  # outside the layer the figure stays home
    action = _action(rig, i, layer)
    start, end = float(layer.start), float(max(layer.end, layer.start + 1))
    looped = layer.path_shape != "LINE" and (layer.path_shape == "CIRCLE" or bool(curve.data.splines and
                                                                                  curve.data.splines[0].use_cyclic_u))
    offset = _fcurve(action, rig, f'constraints["{c.name}"].offset_factor', 0, "Полёт")
    if looped:
        lap = (end - start) / max(layer.laps, 1e-3)
        _keys(offset, [start, start + lap], [0.0, 1.0], cyclic=True, linear=True)
    else:
        _keys(offset, [start, end], [0.0, 1.0], cyclic=False, linear=True)
    influence = _fcurve(action, rig, f'constraints["{c.name}"].influence', 0, "Полёт")
    points = _fades(start, end, layer.fade_in * rate, layer.fade_out * rate, 1.0)
    _keys(influence, [f for f, _ in points], [v for _, v in points], cyclic=False)
    # the action's own frames are the scene's: the strip starts at 0 and holds its ends (home before and after)
    _strip(ad, rig, f"{i + 1} {layer.name}", action, 0.0, end + 1.0, "REPLACE", "HOLD")
    return True
