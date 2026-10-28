"""Can real drones fly it? Speeds, accelerations and distances of the drone points over the animation.

Every Pointille drone point of the scene is followed frame by frame (the figure meshes are not deformed meanwhile:
only the points matter, and a heavy mesh would make it slow). The limits are Skybrush's own safety settings of
the scene when Skybrush is there, its defaults otherwise. Positions are taken at the scene's scale: check the
figure at the size it will have in the sky.

Speeds and accelerations grow with the pace of the animation: a motion too fast is fixed by a longer period
(speeds fall as 1/k, accelerations as 1/k²), so the report says how many times to slow down. Drones too close
are another matter: when they are as close already at rest, the figure is too small for its number of points.
"""

from __future__ import annotations

import json

import bpy
import numpy as np

from . import rig as rig_mod

DEFAULTS = {"distance": 3.0, "xy": 10.0, "up": 2.0, "down": 2.0, "acceleration": 4.0}  # Skybrush's defaults
REPORT_KEY = "dronegen_safety"


def limits(scene) -> dict:
    s = getattr(getattr(scene, "skybrush", None), "safety_check", None)
    if s is None:
        return {**DEFAULTS, "source": "обычные пороги Skybrush (Skybrush в сцене нет)"}
    z = float(s.velocity_z_warning_threshold)
    up = float(s.velocity_z_warning_threshold_up) if getattr(s, "velocity_z_warning_different_up", False) else z
    return {"distance": float(s.proximity_warning_threshold), "xy": float(s.velocity_xy_warning_threshold),
            "up": up, "down": z, "acceleration": float(s.acceleration_warning_threshold),
            "source": "настройки безопасности Skybrush этой сцены"}


def drones(scene) -> list:
    return [o for o in scene.objects if rig_mod.is_points(o)]


def span(rig, scene) -> tuple[int, int]:
    """Frames the animation layers cover (the scene's range when there are none)."""
    rate = scene.render.fps / scene.render.fps_base
    ends = []
    for layer in getattr(rig, "dronegen_layers", []) if rig is not None else []:
        if layer.enabled:
            from . import motion

            once = layer.kind == "MOTION" and (motion.CATALOG.get(layer.motion) or {}).get("kind") == "once"
            ends.append((layer.start, layer.start + layer.period * rate if once else layer.end))
    if not ends:
        return scene.frame_start, scene.frame_end
    return int(min(a for a, _ in ends)), int(np.ceil(max(b for _, b in ends)))


def sample(context, objects: list, frames) -> np.ndarray:
    """World positions of all points of ``objects`` at ``frames``: (frames, drones, 3)."""
    scene = context.scene
    keep = scene.frame_current
    muted = []
    for o in scene.objects:  # the figures' own deformation is not needed here, only the points'
        if o.type == "MESH" and not rig_mod.is_points(o):
            for m in o.modifiers:
                if m.type == "ARMATURE" and m.show_viewport:
                    m.show_viewport = False
                    muted.append(m)
    sizes = [len(o.data.vertices) for o in objects]
    out = np.empty((len(frames), sum(sizes), 3))
    wm = context.window_manager
    wm.progress_begin(0, len(frames))
    try:
        for k, f in enumerate(frames):
            scene.frame_set(int(f))
            depsgraph = context.evaluated_depsgraph_get()
            at = 0
            for o, n in zip(objects, sizes):
                ev = o.evaluated_get(depsgraph)
                co = np.empty(len(ev.data.vertices) * 3)
                ev.data.vertices.foreach_get("co", co)
                m = np.array(ev.matrix_world)
                out[k, at: at + n] = co.reshape(-1, 3)[:n] @ m[:3, :3].T + m[:3, 3]
                at += n
            wm.progress_update(k)
    finally:
        wm.progress_end()
        for m in muted:
            m.show_viewport = True
        scene.frame_set(keep)
    return out


def at_rest(context, objects: list, rig) -> np.ndarray:
    """Positions with the skeleton at rest and its flight paths off: the figure as it was made."""
    data = rig.data if rig is not None else None
    pose = data.pose_position if data is not None else None
    flights = [c for c in (rig.constraints if rig is not None else []) if c.type == "FOLLOW_PATH" and not c.mute]
    try:
        if data is not None:
            data.pose_position = "REST"
        for c in flights:
            c.mute = True
        return sample(context, objects, [context.scene.frame_current])[0]
    finally:
        if data is not None:
            data.pose_position = pose
        for c in flights:
            c.mute = False


def closest(points: np.ndarray) -> tuple[float, int, int]:
    """The smallest distance between two of the points, and which two."""
    n = len(points)
    if n < 2:
        return float("inf"), 0, 0
    best, pair = np.inf, (0, 0)
    p = points.astype(np.float64)
    for a in range(0, n, 512):
        d = ((p[a: a + 512, None, :] - p[None, :, :]) ** 2).sum(-1)
        rows = np.arange(d.shape[0])
        d[rows, a + rows] = np.inf
        k = int(np.argmin(d))
        if d.flat[k] < best:
            best, pair = float(d.flat[k]), (a + k // n, k % n)
    return float(np.sqrt(best)), int(pair[0]), int(pair[1])


def analyze(positions: np.ndarray, frames: list, rate: float, lim: dict, rest: np.ndarray | None) -> dict:
    """Worst speed, climb, descent, acceleration and distance, with the frame and the drone where each happens."""
    dt = (frames[1] - frames[0]) / rate if len(frames) > 1 else 1.0 / rate
    report = {"frames": [int(frames[0]), int(frames[-1])], "drones": int(positions.shape[1]), "limits": lim}
    worst = {}
    if len(frames) > 1:
        v = np.diff(positions, axis=0) / dt
        for key, values in (("xy", np.linalg.norm(v[..., :2], axis=-1)), ("up", v[..., 2]), ("down", -v[..., 2])):
            k = np.unravel_index(int(np.argmax(values)), values.shape)
            worst[key] = {"value": float(max(values[k], 0.0)), "frame": int(frames[k[0] + 1]), "drone": int(k[1])}
        if len(frames) > 2:
            a = np.linalg.norm(np.diff(v, axis=0), axis=-1) / dt
            k = np.unravel_index(int(np.argmax(a)), a.shape)
            worst["acceleration"] = {"value": float(a[k]), "frame": int(frames[k[0] + 1]), "drone": int(k[1])}
    near = [closest(p) for p in positions]
    k = int(np.argmin([d for d, _, _ in near]))
    worst["distance"] = {"value": near[k][0], "frame": int(frames[k]), "drone": near[k][1], "other": near[k][2]}
    report["worst"] = worst
    report["rest_distance"] = closest(rest)[0] if rest is not None else None
    ratios = [worst[key]["value"] / lim[key] for key in ("xy", "up", "down") if key in worst and lim[key] > 0]
    if "acceleration" in worst and lim["acceleration"] > 0:
        ratios.append(np.sqrt(worst["acceleration"]["value"] / lim["acceleration"]))
    report["slow"] = float(max(ratios + [1.0]))
    return report


def part(index: int, objects: list) -> str:
    """Which part of which figure a drone sits on: by the bone its point mostly follows."""
    for o in objects:
        n = len(o.data.vertices)
        if index < n:
            if o.get("dronegen_effect_name"):
                return f"эффект «{o['dronegen_effect_name']}», дрон {index + 1}"
            v = o.data.vertices[index]
            names = {g.index: g.name for g in o.vertex_groups}
            best = max(((g.weight, names.get(g.group, "")) for g in v.groups
                        if not names.get(g.group, "").startswith(("Pointille", "Skybrush", rig_mod.PART_PREFIX))),
                       default=(0, ""))[1]
            rig = next((m.object for m in o.modifiers if m.type == "ARMATURE" and m.object is not None), None)
            bone = rig.data.bones.get(best) if rig is not None and best else None
            if bone is None:
                return f"«{o.name}», точка {index}"
            role = bone.get("dg_role", "extra")
            return rig_mod.part_title(role, bone.get("dg_side", "")) + f" ({bone.name})"
        index -= n
    return "?"


def problems(report: dict, objects: list) -> list[str]:
    """What is wrong, in words, worst first."""
    lim, worst, out = report["limits"], report["worst"], []
    words = {"xy": "Скорость по горизонтали", "up": "Подъём", "down": "Спуск", "acceleration": "Ускорение"}
    for key, word in words.items():
        w = worst.get(key)
        if w and w["value"] > lim[key]:
            unit = "м/с²" if key == "acceleration" else "м/с"
            out.append(f"{word} {w['value']:.1f} {unit} при пределе {lim[key]:g} — кадр {w['frame']}, "
                       f"{part(w['drone'], objects)}")
    d, rest = worst["distance"], report.get("rest_distance")
    if d["value"] < lim["distance"]:
        if rest is not None and rest < lim["distance"]:
            out.append(f"Дроны ближе {lim['distance']:g} м уже в покое ({rest:.2f} м): фигура мала для своего числа "
                       f"точек — увеличьте её (Pointille «Масштабировать») или возьмите меньше точек")
        else:
            out.append(f"Анимация сближает дронов до {d['value']:.2f} м (предел {lim['distance']:g}) — кадр "
                       f"{d['frame']}, {part(d['drone'], objects)} и {part(d['other'], objects)}")
    return out


def check(context, rig) -> dict:
    """Follow every drone point over the animation's frames and judge it; the report is kept on the rig."""
    scene = context.scene
    objects = drones(scene)
    if not objects:
        raise RuntimeError("В сцене нет точек дронов: сначала Pointille «Распределить вершины»")
    first, last = span(rig, scene)
    frames = list(range(first, last + 1))
    rate = scene.render.fps / scene.render.fps_base
    lim = limits(scene)
    report = analyze(sample(context, objects, frames), frames, rate, lim, at_rest(context, objects, rig))
    report["problems"] = problems(report, objects)
    report["ok"] = not report["problems"]
    if rig is not None:
        rig[REPORT_KEY] = json.dumps(report, ensure_ascii=False)
    return report


def last(rig) -> dict | None:
    try:
        return json.loads(rig.get(REPORT_KEY)) if rig is not None and rig.get(REPORT_KEY) else None
    except ValueError:
        return None


def slow_down(rig, factor: float) -> None:
    """Every layer ``factor`` times slower: longer periods and fades, fewer laps of a flight."""
    for layer in rig.dronegen_layers:
        layer.fade_in *= factor
        layer.fade_out *= factor
        if layer.kind == "PATH":
            layer.laps = max(0.05, layer.laps / factor)
        else:
            layer.period *= factor
