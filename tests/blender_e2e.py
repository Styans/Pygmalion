"""Runs inside headless Blender (see run_blender_e2e.py). Every check is recorded in a JSON report.

Timers do not fire while a background script runs, so the test pumps ``jobs.tick()`` itself:
the same function the add-on's timer calls.
"""

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import addon_utils
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--dead-port", type=int, required=True)
ap.add_argument("--real", action="store_true")
args = ap.parse_args(argv)

MOD = "bl_ext.user_default.dronegen"
results = []


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            info = fn()
            results.append({"name": name, "ok": True, "seconds": round(time.time() - t0, 2), "info": info or ""})
        except Exception as e:  # noqa: BLE001 - record and go on with the next check
            traceback.print_exc()
            results.append({"name": name, "ok": False, "seconds": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"})
        return fn

    return deco


if MOD not in bpy.context.preferences.addons:
    addon_utils.enable(MOD, default_set=True)
jobs = importlib.import_module(MOD + ".jobs")
gallery = importlib.import_module(MOD + ".gallery")
presets = importlib.import_module(MOD + ".presets")
ui = importlib.import_module(MOD + ".ui")
prefs = bpy.context.preferences.addons[MOD].preferences
prefs.server_url = args.url
prefs.cache_dir = args.cache
scene = bpy.context.scene
s = scene.dronegen


def pump(cond, timeout, what="задача"):
    t0 = time.time()
    while not cond():
        jobs.tick()
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{what} не завершилась за {timeout:.0f} с")
        time.sleep(0.05)
    jobs.tick()


def run_job(op, kind, timeout=120):
    before = {j.key for j in jobs.jobs_of(kind)}
    assert op() == {"FINISHED"}
    job = next(j for j in jobs.jobs_of(kind) if j.key not in before)
    pump(lambda: not job.active, timeout, job.title)
    return job


def expect_error(op, text):
    """An operator that refuses: Blender raises RuntimeError with the report text."""
    try:
        result = op()
    except RuntimeError as e:
        assert text in str(e), str(e)
        return
    assert result == {"CANCELLED"}, result


def figure_mask(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path), check_existing=True)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(h, w, 4)[::-1, :, :3]  # row 0 = top
    mask = np.linalg.norm(rgb - 1.0, axis=-1) > 0.15
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def front_iou(obj, image_path: Path, n=96) -> float:
    """Rays from the Front view (along +Y) against the picture's figure."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    corners = np.array([obj.matrix_world @ Vector(c) for c in obj.bound_box])
    (x0, y0, z0), (x1, _, z1) = corners.min(0), corners.max(0)
    want = figure_mask(image_path)
    k = n / max(want.shape)
    h, w = max(1, round(want.shape[0] * k)), max(1, round(want.shape[1] * k))
    rows = (np.arange(h) * want.shape[0] / h).astype(int)
    cols = (np.arange(w) * want.shape[1] / w).astype(int)
    want = want[rows][:, cols]
    got = np.zeros((h, w), bool)
    for r in range(h):
        z = z1 - (r + 0.5) / h * (z1 - z0)
        for c in range(w):
            x = x0 + (c + 0.5) / w * (x1 - x0)
            got[r, c] = bvh.ray_cast(Vector((x, y0 - 1.0, z)), Vector((0.0, 1.0, 0.0)))[0] is not None
    return float((got & want).sum() / max(1, (got | want).sum()))


def check_mesh(obj, size):
    assert obj.users_collection[0].name == "DroneGen", [c.name for c in obj.users_collection]
    assert bpy.context.view_layer.objects.active == obj and obj.select_get()
    assert tuple(obj.location) == (0.0, 0.0, 0.0) and tuple(obj.scale) == (1.0, 1.0, 1.0)
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    assert abs((co.max(0) - co.min(0)).max() - size) < 1e-3, co.max(0) - co.min(0)
    assert np.allclose((co.min(0) + co.max(0)) / 2, 0.0, atol=1e-4)
    assert abs(max(obj.dimensions) - size) < 1e-3, tuple(obj.dimensions)  # what the user sees in N-panel
    assert len(obj.data.color_attributes) >= 1
    assert obj.data.materials[0].name == "DroneGen Vertex Color"


# -- drawing without a window: a recording layout that validates names against RNA -------------
ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())


class FakeOp:
    def __init__(self, idname):
        mod, name = idname.split(".")
        self._props = getattr(getattr(bpy.ops, mod), name).get_rna_type().properties.keys()

    def __setattr__(self, key, value):
        if key != "_props" and key not in self._props:
            raise AttributeError(f"у оператора нет свойства {key}")
        object.__setattr__(self, key, value)


class FakeLayout:
    def __init__(self, log):
        self.log = log

    def _icon(self, kw):
        icon = kw.get("icon")
        if icon is not None and icon not in ICONS:
            raise ValueError(f"нет иконки {icon}")

    def prop(self, data, name, **kw):
        if name not in data.bl_rna.properties:
            raise AttributeError(f"{data.bl_rna.identifier} не имеет свойства {name}")
        self._icon(kw)
        self.log.append(("prop", name))

    def operator(self, idname, **kw):
        self._icon(kw)
        self.log.append(("operator", idname))
        return FakeOp(idname)

    def label(self, text="", **kw):
        self._icon(kw)
        self.log.append(("label", text))

    def prop_enum(self, data, name, value, **kw):
        prop = data.bl_rna.properties[name]
        if value not in prop.enum_items.keys():
            raise ValueError(f"{name}: нет значения {value}")
        self._icon(kw)
        self.log.append(("prop_enum", f"{name}={value}"))

    def menu(self, menu, **kw):
        getattr(bpy.types, menu)  # registered
        self._icon(kw)
        self.log.append(("menu", menu))

    def panel_prop(self, data, name):
        if name not in data.bl_rna.properties:
            raise AttributeError(f"{data.bl_rna.identifier} не имеет свойства {name}")
        self.log.append(("fold", name))
        return self, self  # open: its contents are drawn and checked too

    def progress(self, factor=0.0, type="BAR", text=""):
        assert 0.0 <= factor <= 1.0
        self.log.append(("progress", text))

    def template_icon_view(self, data, name, **kw):
        if name not in data.bl_rna.properties:
            raise AttributeError(name)
        self.log.append(("gallery", name))

    def template_list(self, list_type, list_id, data, prop, active_data, active_prop, **kw):
        getattr(bpy.types, list_type)  # registered
        for name, owner in ((prop, data), (active_prop, active_data)):
            if name not in owner.bl_rna.properties:
                raise AttributeError(f"{owner.bl_rna.identifier} не имеет свойства {name}")
        self.log.append(("list", prop))

    def template_icon(self, icon_value=0, scale=1.0):
        self.log.append(("icon", icon_value))

    def separator(self, **kw):
        pass

    def row(self, **kw):
        return self

    column = box = grid_flow = row


class FakeRegion:
    width = 320


class FakeContext:
    def __init__(self):
        self.scene = bpy.context.scene
        self.view_layer = bpy.context.view_layer
        self.window_manager = bpy.context.window_manager
        self.preferences = bpy.context.preferences
        self.region = FakeRegion()
        self.active_object = bpy.context.view_layer.objects.active
        self.mode = bpy.context.mode


def draw_panels() -> list:
    log = []
    for cls in (ui.DRONEGEN_PT_main, ui.DRONEGEN_PT_image, ui.DRONEGEN_PT_gallery, ui.DRONEGEN_PT_mesh, ui.DRONEGEN_PT_anim,
                ui.DRONEGEN_PT_layers, ui.DRONEGEN_PT_drones_check):
        if hasattr(cls, "poll") and not cls.poll(FakeContext()):
            continue
        panel = type("P", (), {"layout": FakeLayout(log)})()
        for method in ("draw_header", "draw_header_preset", "draw"):
            if hasattr(cls, method):
                getattr(cls, method)(panel, FakeContext())
    menu = type("M", (), {"layout": FakeLayout(log)})()
    ui.DRONEGEN_MT_presets.draw(menu, FakeContext())
    return log


# -- checks -----------------------------------------------------------------------------------------
@check("панель рисуется без сервера")
def _():
    jobs.STATE.server.update(ok=False, offline=True, message="Нет связи с сервером")
    log = draw_panels()
    assert ("operator", "dronegen.start_server") in log
    return f"{len(log)} элементов"


@check("длинный текст переносится, а не обрезается")
def _():
    text = ("Нет связи с сервером http://127.0.0.1:8188. Запустите сервер (кнопка «Запустить сервер» "
            "или start_server.bat) и проверьте адрес в настройках аддона.")
    lines = ui.text_lines(text, 200, bpy.context)
    assert len(lines) > 2 and " ".join(lines).split() == text.split(), lines
    return f"{len(lines)} строк"


@check("связь с сервером")
def _():
    assert bpy.ops.dronegen.check_server() == {"FINISHED"}
    pump(lambda: jobs.STATE.server["ok"] is not None and "Проверяю" not in jobs.STATE.server["message"], 20, "проверка")
    assert jobs.STATE.server["ok"], jobs.STATE.server["message"]
    return jobs.STATE.server["message"]


@check("пресет и стиль картинки")
def _():
    assert bpy.ops.dronegen.use_preset(key="SHANYRAK") == {"FINISHED"}
    assert "shanyrak" in s.prompt
    s.image_backend = "mock"
    assert s.steps == 8  # the default of the chosen model
    s.image_style = "sheet"  # a style picks the picture size and the 3D method that suit it
    assert (s.size, s.mesh_backend, s.style()) == ("1920x576", "hunyuan_mv", "sheet")
    s.image_style = "none"
    assert s.style() is None
    s.image_style = "flat"  # the server adds the style words (checked with the mock model below)
    assert (s.size, s.mesh_backend, s.style()) == ("768x768", "silhouette", "flat")


@check("пустой промпт → понятная ошибка")
def _():
    keep, s.prompt = s.prompt, "   "
    expect_error(bpy.ops.dronegen.generate_image, "Введите промпт")
    assert any("Введите промпт" in e for e in jobs.STATE.errors)
    s.prompt = keep
    bpy.ops.dronegen.dismiss()


image_path = None


@check("промпт → картинка в галерее")
def _():
    global image_path
    s.seed, s.size = 2, "512x512"
    job = run_job(bpy.ops.dronegen.generate_image, "txt2img")
    assert job.status == "done", job.error
    image_path = gallery.selected(bpy.context)
    assert image_path is not None and image_path.is_file()
    meta = gallery.meta(image_path)
    assert meta["seed"] == 2 and meta["width"] == 512
    assert meta["style"] == "flat" and meta["prompt"].endswith("flat colors")  # style words added by the server
    log = draw_panels()
    assert ("gallery", "dronegen_gallery") in log and any(k == "icon" for k, _ in log)
    if not bpy.app.background:  # previews get icon ids only with a UI
        assert gallery.icon_id(image_path.name) > 0
    return image_path.name


@check("галерея: выбор держится за файл, листание, удаление")
def _():
    s.seed = 3
    job = run_job(bpy.ops.dronegen.generate_image, "txt2img")
    assert job.status == "done", job.error
    second = gallery.selected(bpy.context)
    assert second != image_path and gallery.position(second.name) == (1, len(gallery.names()))  # the new one goes first
    assert bpy.ops.dronegen.gallery_step(delta=1) == {"FINISHED"}
    assert gallery.selected(bpy.context) == image_path
    gallery.refresh()  # a re-read keeps the selection
    assert gallery.selected(bpy.context) == image_path
    bpy.ops.dronegen.gallery_step(delta=-1)
    assert gallery.selected(bpy.context) == second
    gallery.remove(second.name)
    assert not second.exists() and gallery.selected(bpy.context) == image_path  # its neighbour takes the place
    return f"{len(gallery.names())} картинок"


@check("Открыть в Image Editor")
def _():
    assert bpy.ops.dronegen.open_image_editor() == {"FINISHED"}
    assert any(Path(bpy.path.abspath(i.filepath)).name == image_path.name for i in bpy.data.images)


@check("картинка → силуэт → меш в сцене")
def _():
    s.source, s.mesh_backend, s.remove_bg, s.target_size = "GALLERY", "silhouette", True, 2.0
    assert s.mc_resolution == "192"  # the default of the chosen backend
    job = run_job(bpy.ops.dronegen.make_3d, "img2mesh", timeout=180)
    assert job.status == "done", job.error
    obj = bpy.data.objects[job.output]
    check_mesh(obj, 2.0)
    score = front_iou(obj, image_path)
    assert score > 0.8, score
    return f"{obj.name}: {len(obj.data.polygons)} полигонов, IoU вида спереди {score:.2f}"


@check("картинка → тестовый меш, другой размер")
def _():
    s.mesh_backend, s.target_size = "mock", 5.0
    job = run_job(bpy.ops.dronegen.make_3d, "img2mesh")
    assert job.status == "done", job.error
    check_mesh(bpy.data.objects[job.output], 5.0)


def creature():
    """A horse-like figure of primitives facing -Y (body, four legs, a neck with a head, a tail) and Pointille-like
    drone points on it: what a generated model and its points look like to the skeleton."""
    from mathutils import Vector

    parts = []

    def capsule(a, b, r):
        a, b = Vector(a), Vector(b)
        d = b - a
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=r, depth=d.length, location=(a + b) / 2,
                                            rotation=d.to_track_quat("Z", "Y").to_euler())
        parts.append(bpy.context.active_object)
        for p in (a, b):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=r, location=p)
            parts.append(bpy.context.active_object)

    capsule((0, 0.6, 1.0), (0, -0.6, 1.0), 0.3)
    for x in (-0.2, 0.2):
        for y in (-0.45, 0.45):
            capsule((x, y, 0.9), (x, y, 0.06), 0.08)
    capsule((0, -0.55, 1.1), (0, -0.85, 1.65), 0.15)
    capsule((0, -0.85, 1.65), (0, -1.25, 1.55), 0.13)
    capsule((0, 0.75, 1.0), (0, 1.25, 0.55), 0.045)
    for o in bpy.context.view_layer.objects:
        o.select_set(o in parts)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    horse = bpy.context.active_object
    horse.name = "DG Тестовый конь"
    me = bpy.data.meshes.new("points")  # drones: a point on each leg low down, on the head and on the body
    me.from_pydata([(0.2, -0.45, 0.3), (-0.2, -0.45, 0.3), (0.2, 0.45, 0.3), (-0.2, 0.45, 0.3), (0, -1.2, 1.6),
                    (0, 0, 1.25)], [], [])
    points = bpy.data.objects.new("DG Тестовый конь_points", me)
    bpy.context.scene.collection.objects.link(points)
    points.parent = horse  # the joined mesh keeps the first part's turn: the points stay where they were put
    points.matrix_parent_inverse = horse.matrix_world.inverted()
    points["pointille_drones"] = 1
    points.vertex_groups.new(name="Pointille").add(list(range(6)), 1.0, "REPLACE")
    return horse, points


@check("скелет: модель → кости → меш и точки дронов двигаются с ними")
def _():
    horse, points = creature()
    rig_mod = importlib.import_module(MOD + ".rig")
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = points  # the points are enough: the figure behind them is found
    s.rig_plan, s.rig_method, s.rig_bind = "quadruped", "template", True
    job = run_job(bpy.ops.dronegen.build_rig, "rig", timeout=120)
    assert job.status == "done", job.error
    rig = rig_mod.rig_of(horse)
    assert rig is not None and horse.parent is rig and rig_mod.bound(horse, rig) and rig_mod.bound(points, rig)
    roles = {b.get("dg_role") for b in rig.data.bones}
    assert {"leg_front", "leg_hind", "spine", "neck", "head", "tail", "root"} <= roles, roles
    found = rig_mod.describe({**rig_mod.info(rig), "bones": [{"role": b.get("dg_role")} for b in rig.data.bones]})
    assert "4 ноги" in found and "хвост" in found and job.output == "Скелет готов и привязан с точками дронов", (found, job.output)
    assert points.modifiers[0].type == "ARMATURE"
    # light by parts: a group per part of the body, which part and how far along it each drone is
    groups = {vg.name for vg in points.vertex_groups if vg.name.startswith(rig_mod.PART_PREFIX)}
    assert {"DG· передняя нога (левое)", "DG· задняя нога (правое)"} <= groups and groups & {"DG· голова", "DG· шея"}, groups
    along = np.empty(len(points.data.vertices), np.float32)
    points.data.attributes["dg_along"].data.foreach_get("value", along)
    part = np.empty(len(points.data.vertices), np.int32)
    points.data.attributes["dg_part"].data.foreach_get("value", part)
    titles = json.loads(points[rig_mod.PARTS_KEY])
    assert titles[part[0]] == "DG· передняя нога (левое)" and 0.4 < along[0] <= 1.0, (titles, part, along)
    assert ((along >= 0) & (along <= 1)).all()
    leg = next(b.name for b in rig.data.bones if b.get("dg_chain") == "leg_front.L" and b.get("dg_segment") == 1)

    def drones():
        ev = points.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return np.array([tuple(points.matrix_world @ v.co) for v in ev.data.vertices])

    before = drones()
    pb = rig.pose.bones[leg]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.8, 0, 0)
    bpy.context.view_layer.update()
    moved = np.linalg.norm(drones() - before, axis=1)
    assert moved[0] > 0.1, moved  # the point on the front left leg (+X: the figure's left, it faces -Y)
    assert moved[[1, 2, 3, 4]].max() < 0.02, moved  # the other legs and the head stay
    assert ("operator", "dronegen.bind_rig") in draw_panels()
    assert bpy.ops.dronegen.reset_pose() == {"FINISHED"}
    bpy.context.view_layer.update()
    assert np.allclose(drones(), before, atol=1e-5)
    assert bpy.ops.dronegen.bind_rig() == {"FINISHED"}  # binding again keeps one modifier and the drones' groups
    assert len([m for m in points.modifiers if m.type == "ARMATURE"]) == 1 and "Pointille" in points.vertex_groups
    rig_mod.remove(horse, rig)
    assert rig_mod.rig_of(horse) is None and not any(m.type == "ARMATURE" for m in points.modifiers)
    assert "dg_along" not in points.data.attributes and not any(vg.name.startswith("DG· ") for vg in points.vertex_groups)
    return f"{found}; кость {leg} подняла свою ногу на {moved[0]:.2f}"


@check("анимация: слои → дорожки NLA, полёт, проверка дронов")
def _():
    rig_mod = importlib.import_module(MOD + ".rig")
    motion_mod = importlib.import_module(MOD + ".motion")
    horse, points = creature()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = horse
    s.rig_plan, s.rig_method, s.rig_bind = "quadruped", "template", True
    job = run_job(bpy.ops.dronegen.build_rig, "rig", timeout=120)
    assert job.status == "done", job.error
    rig = rig_mod.rig_of(horse)
    scene = bpy.context.scene
    scene.frame_current = 1
    assert bpy.ops.dronegen.layer_add(kind="MOTION", motion="gallop") == {"FINISHED"}
    assert "gallop" in motion_mod.CATALOG and "wing_flap" not in {k for k, _ in motion_mod.fitting(rig)}  # no wings
    gallop = rig.dronegen_layers[0]
    assert gallop.name == "Галоп" and gallop.period == motion_mod.CATALOG["gallop"]["period"]
    gallop.start, gallop.end = 1, 121
    assert bpy.ops.dronegen.layer_add(kind="PATH") == {"FINISHED"}
    flight = rig.dronegen_layers[1]
    flight.start, flight.end, flight.radius, flight.fade_in, flight.fade_out = 1, 121, 10.0, 1.0, 1.0
    assert bpy.ops.dronegen.layer_add(kind="MOTION") == {"FINISHED"}
    assert bpy.ops.dronegen.set_motion(key="tail_wave") == {"FINISHED"} and rig.dronegen_layers[2].motion == "tail_wave"
    log = draw_panels()
    assert ("list", "dronegen_layers") in log and ("operator", "dronegen.apply_layers") in log
    job = run_job(bpy.ops.dronegen.apply_layers, "anim", timeout=60)
    assert job.status == "done", job.error
    tracks = [t.name for t in rig.animation_data.nla_tracks]
    assert len([t for t in tracks if t.startswith("DG· ")]) == 3, tracks
    assert any(c.type == "FOLLOW_PATH" for c in rig.constraints)

    def drones(frame):
        scene.frame_set(frame)
        ev = points.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return np.array([tuple(points.matrix_world @ v.co) for v in ev.data.vertices])

    home = drones(1)
    mid = drones(61)
    assert np.linalg.norm(rig.matrix_world.translation) > 5.0, tuple(rig.matrix_world.translation)  # out on the circle
    assert np.allclose(drones(130), home, atol=1e-4)  # after the layers: home, at rest
    legs = drones(61) - mid.mean(axis=0) - (home - home.mean(axis=0))
    assert np.abs(legs[:4]).max() > 0.05  # the legs move relative to the body (the gallop), not only the flight
    assert bpy.ops.dronegen.check_drones() == {"FINISHED"}
    safety_mod = importlib.import_module(MOD + ".safety")
    report = safety_mod.last(rig)
    every = sum(len(o.data.vertices) for o in safety_mod.drones(scene))  # all drones of the scene, both horses
    last = max(layer.end for layer in rig.dronegen_layers)  # the tail wave lasts its default 10 s
    assert report["drones"] == every >= 6 and report["frames"] == [1, last] and report["worst"]["xy"]["value"] > 0, report
    assert ("operator", "dronegen.check_drones") in draw_panels()
    period = gallop.period
    if report["slow"] > 1.05:
        before = {j.key for j in jobs.jobs_of("anim")}
        assert bpy.ops.dronegen.slow_down(factor=2.0) == {"FINISHED"}
        again = next(j for j in jobs.jobs_of("anim") if j.key not in before)
        pump(lambda: not again.active, 60, "сборка анимации")
        assert again.status == "done" and gallop.period == 2 * period
    assert bpy.ops.dronegen.clear_layers() == {"FINISHED"}
    assert not [t for t in rig.animation_data.nla_tracks if t.name.startswith("DG· ")] and not rig.constraints
    assert len(rig.dronegen_layers) == 3  # the layers stay, only what was built goes
    return f"{job.output}; быстрее всего {report['worst']['xy']['value']:.1f} м/с, замедлить в {report['slow']:.1f} раза"


@check("эффекты: волна на точках, звезда со своими дронами из резерва, след копыта")
def _():
    rig_mod = importlib.import_module(MOD + ".rig")
    fx = importlib.import_module(MOD + ".fx")
    horse, points = creature()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = horse
    s.rig_plan, s.rig_method, s.rig_bind = "quadruped", "template", True
    job = run_job(bpy.ops.dronegen.build_rig, "rig", timeout=120)
    assert job.status == "done", job.error
    rig = rig_mod.rig_of(horse)
    scene = bpy.context.scene
    scene.frame_current = 1
    assert bpy.ops.dronegen.layer_add(kind="MOTION", motion="gallop") == {"FINISHED"}
    rig.dronegen_layers[0].start, rig.dronegen_layers[0].end = 1, 200
    assert bpy.ops.dronegen.layer_add(kind="SHAPE") == {"FINISHED"}
    wave = rig.dronegen_layers[1]
    wave.start, wave.end, wave.fx_size = 1, 200, 0.3
    scene.frame_current = 200  # late enough for the drones to fly out of the reserve after the show begins
    assert bpy.ops.dronegen.layer_add(kind="EFFECT") == {"FINISHED"}
    star = rig.dronegen_layers[2]
    assert star.effect == "star" and star.start == 200 and star.bone, (star.effect, star.start, star.bone)
    star.size, star.end = 12.0, 260
    assert bpy.ops.dronegen.layer_add(kind="EFFECT") == {"FINISHED"}
    trail = rig.dronegen_layers[3]
    trail.effect, trail.name, trail.bone, trail.end = "trail", "След", "leg_front.L", 130
    assert ("operator", "dronegen.apply_layers") in draw_panels()  # every kind of layer draws
    job = run_job(bpy.ops.dronegen.apply_layers, "anim", timeout=60)
    assert job.status == "done", job.error
    assert "эффектов формы: 1" in job.output and "эффектов с дронами: 1" in job.output, job.output
    pump(lambda: not any(j.active for j in jobs.jobs_of("trail")), 60, "след")
    done = [j for j in jobs.jobs_of("trail") if j.meta.get("rig") == rig.name]
    assert done and done[-1].status == "done", [(j.status, j.error) for j in done]
    made = {o.get("dronegen_effect_name"): o for o in fx.effects_of(rig)}
    assert set(made) == {star.name, "След"}, list(made)
    shape_mod = next(m for m in points.modifiers if m.name.startswith("DG· "))
    assert shape_mod.type == "NODES" and points.modifiers[0].type == "ARMATURE"

    def local(obj, frame):
        scene.frame_set(frame)
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        co = np.empty(len(ev.data.vertices) * 3)
        ev.data.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)

    def attr(obj, name):
        a = obj.data.attributes[name]
        out = np.empty(len(a.data) * 3)
        a.data.foreach_get("vector", out)
        return out.reshape(-1, 3)

    shown = local(points, 60)
    shape_mod.show_viewport = False
    plain = local(points, 60)
    shape_mod.show_viewport = True
    assert np.abs(shown - plain).max() > 0.01  # the wave moves the drones beyond what the skeleton does
    obj = made[star.name]
    n = len(obj.data.vertices)
    assert n > 5 and obj.parent is rig and "Pointille" in obj.vertex_groups and obj.get("pointille_drones")
    assert np.allclose(local(obj, 1), attr(obj, "dg_reserve"), atol=1e-3), job.warnings  # waiting in the reserve
    assert np.allclose(local(obj, 230), attr(obj, "dg_shape"), atol=1e-3)  # the star in the sky
    assert len(local(obj, 400)) == n  # the drones stay the same drones all along
    sky = local(obj, 230) @ np.array(obj.matrix_world)[:3, :3].T
    assert 8.0 < np.ptp(sky, axis=0).max() < 13.0, np.ptp(sky, axis=0)  # 12 m across in the sky
    budget = fx.budget(rig, scene)
    assert budget["effects"] == sum(len(o.data.vertices) for o in made.values()) and budget["figure"] == 6, budget
    assert ("operator", "dronegen.check_drones") in draw_panels()
    assert bpy.ops.dronegen.clear_layers() == {"FINISHED"}
    assert not fx.effects_of(rig) and not [m for m in points.modifiers if m.name.startswith("DG· ")]
    return f"{job.output}; звезда {n} дронов, бюджет {budget}"


@check("ошибка: бэкенд недоступен")
def _():
    s.mesh_backend = "triposr" if not args.real else "mock"
    if args.real:
        return "пропущено в режиме --real"
    job = run_job(bpy.ops.dronegen.make_3d, "img2mesh")
    assert job.status == "error" and "triposr" in job.error, job.error
    assert any("triposr" in e for e in jobs.STATE.errors)
    log = draw_panels()
    assert ("operator", "dronegen.dismiss") in log
    bpy.ops.dronegen.dismiss()
    return job.error


@check("ошибка: файл не картинка")
def _():
    bad = Path(args.cache) / "not_an_image.png"
    bad.write_bytes(b"this is text, not a picture")
    s.source, s.image_file, s.mesh_backend = "FILE", str(bad), "silhouette"
    job = run_job(bpy.ops.dronegen.make_3d, "img2mesh")
    s.source = "GALLERY"
    assert job.status == "error" and "картинк" in job.error, job.error
    bpy.ops.dronegen.dismiss()
    return job.error


@check("ошибка: сервер не запущен")
def _():
    prefs.server_url = f"http://127.0.0.1:{args.dead_port}"
    try:
        job = run_job(bpy.ops.dronegen.generate_image, "txt2img", timeout=30)
        assert job.status == "error" and "Нет связи" in job.error, job.error
        bpy.ops.dronegen.check_server()
        pump(lambda: jobs.STATE.server["ok"] is False, 20, "проверка")
        assert jobs.STATE.server["offline"], jobs.STATE.server  # «not running», not some other failure
        log = draw_panels()
        assert ("operator", "dronegen.start_server") in log
    finally:
        prefs.server_url = args.url
        bpy.ops.dronegen.dismiss()
    return job.error


@check("отмена задачи")
def _():
    s.image_backend, s.steps = "mock", 100
    before = {j.key for j in jobs.jobs_of("txt2img")}
    bpy.ops.dronegen.generate_image()
    job = next(j for j in jobs.jobs_of("txt2img") if j.key not in before)
    pump(lambda: job.status == "running" and job.progress > 0, 30, "запуск")
    assert "progress" in [k for k, _ in draw_panels()]
    bpy.ops.dronegen.cancel_job(key=job.key)
    pump(lambda: not job.active, 30, "отмена")
    assert job.status == "cancelled", job.status
    s.steps = 8


link = importlib.import_module(MOD + ".link")
studio = importlib.import_module(MOD + ".studio")
workspace = importlib.import_module(MOD + ".workspace")


def api(method, path, body=None):
    import urllib.request

    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(args.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


@check("студия: модель из очереди приходит в сцену")
def _():
    # a mesh made by the server (the mock backend), as the Studio's «В Blender» button would send it
    s.mesh_backend, s.source, s.target_size = "mock", "GALLERY", 2.0
    job = run_job(bpy.ops.dronegen.make_3d, "img2mesh")
    assert job.status == "done", job.error
    server_name = Path(job.files["mesh"]).name
    project = api("POST", "/studio/api/projects", {"title": "Тест Blender"})
    chat = api("POST", "/studio/api/chats", {"project_id": project["id"]})
    item = api("POST", f"/studio/api/chats/{chat['id']}/blender", {"file": server_name, "name": "Тест студии"})
    assert item["status"] == "queued"
    before = {o.name for o in bpy.data.objects}
    link._state["next"] = 0.0
    t0 = time.time()
    while not any(o.name.startswith("DG Тест студии") for o in bpy.data.objects):
        link.tick()
        assert time.time() - t0 < 30, "модель из студии не пришла в сцену"
        time.sleep(0.1)
    obj = next(o for o in bpy.data.objects if o.name not in before)
    check_mesh(obj, 2.0)
    assert obj["dronegen_source"] == "studio" and obj["dronegen_inbox"] == item["id"]
    deadline = time.time() + 10
    while True:  # the report back to the server runs in a thread
        done = [i for i in api("GET", "/blender/inbox?status=imported") if i["id"] == item["id"]]
        if done or time.time() > deadline:
            break
        time.sleep(0.2)
    assert done and done[0]["object"] == obj.name, done
    assert api("GET", "/blender/inbox") == []  # nothing left to take
    return obj.name


@check("студия: сломанный файл → ошибка в чате, Blender жив")
def _():
    project = api("GET", "/studio/api/projects")[0]
    chat = api("POST", "/studio/api/chats", {"project_id": project["id"]})
    out = Path(args.cache).parent / "server" / "outputs"  # the server's outputs folder (set up by the runner)
    broken = out / "broken_model.glb"
    broken.write_bytes(b"glTF this is not a model")
    item = api("POST", f"/studio/api/chats/{chat['id']}/blender", {"file": "broken_model.glb", "name": "Битый"})
    link._state["next"] = 0.0
    t0 = time.time()
    while not any(i["id"] == item["id"] for i in api("GET", "/blender/inbox?status=failed")):
        link.tick()
        assert time.time() - t0 < 30, "ошибка не дошла до сервера"
        time.sleep(0.1)
    assert any("Битый" in e for e in jobs.STATE.errors), jobs.STATE.errors
    bpy.ops.dronegen.dismiss()


@check("студия: окно открывается, сервер запускается при необходимости")
def _():
    opened = []
    real_open = studio.open_window
    studio.open_window = lambda url, profile: opened.append((url, str(profile))) or "тест"
    try:
        # the server is up: only the window opens
        studio.open_when_ready(args.url, "", 5.0, "", Path(args.cache) / "profile", wait=5)
        assert opened and opened[0][0] == args.url + "/studio/", opened
        assert jobs.tick() >= 0 and jobs.STATE.server["ok"] is True and "открыта" in jobs.STATE.server["message"]
        # nothing answers and no server folder is set: a readable reason instead of a hang
        opened.clear()
        studio.open_when_ready(f"http://127.0.0.1:{args.dead_port}", "", 2.0, "", Path(args.cache) / "profile", wait=5)
        jobs.tick()
        assert not opened and jobs.STATE.server["ok"] is False and "Папка сервера" in jobs.STATE.server["message"], jobs.STATE.server
        # the token travels in the address for the event streams
        from importlib import import_module

        client_mod = import_module(MOD + ".client")
        assert client_mod.Client("http://127.0.0.1:8188", "a b").studio_url() == "http://127.0.0.1:8188/studio/?token=a%20b"
        # the operator itself: under way, no error
        assert bpy.ops.dronegen.open_studio() == {"FINISHED"}
    finally:
        studio.open_window = real_open
    jobs.STATE.server.update(ok=True, message="проверено")


@check("студия: горячая клавиша Alt+Shift+D")
def _():
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is None:
        return "в фоновом режиме нет раскладки клавиш"
    items = [(km.name, kmi.type, kmi.alt, kmi.shift) for km in kc.keymaps for kmi in km.keymap_items
             if kmi.idname == "dronegen.open_studio"]
    assert items == [("Window", "D", True, True)], items
    return str(items)


@check("воркспейс DroneGen собирается и подключается")
def _():
    template = Path(args.cache) / "workspace-test.blend"
    workspace.build_template(template)
    assert template.is_file() and template.stat().st_size > 10_000
    mode_before = bpy.context.object.mode if bpy.context.object else None
    assert "DroneGen" not in [w.name for w in bpy.data.workspaces]
    assert workspace.activate(bpy.context, template)
    ws = bpy.data.workspaces["DroneGen"]
    kinds = sorted(a.type for a in ws.screens[0].areas)
    assert {"IMAGE_EDITOR", "VIEW_3D", "OUTLINER", "PROPERTIES"} <= set(kinds), kinds
    assert ws.object_mode == "OBJECT"
    assert (bpy.context.object.mode if bpy.context.object else None) == mode_before  # the scene is untouched
    # (switching the window to it happens in the event loop: checked in the GUI smoke test)
    # a second call only switches
    assert workspace.activate(bpy.context, template) and len([w for w in bpy.data.workspaces if w.name.startswith("DroneGen")]) == 1
    return str(kinds)


@check("аддон выключается и включается")
def _():
    global jobs, gallery, ui, s, prefs
    addon_utils.disable(MOD, default_set=False)
    assert not hasattr(bpy.types.Scene, "dronegen")
    addon_utils.enable(MOD, default_set=False)
    # RNA objects taken before re-registration are dangling now (writing to them crashes Blender):
    # take everything again, as the add-on itself does on every call
    s = bpy.context.scene.dronegen
    assert s.prompt  # the settings live in the scene and survive
    prefs = bpy.context.preferences.addons[MOD].preferences
    prefs.server_url, prefs.cache_dir = args.url, args.cache
    jobs = importlib.import_module(MOD + ".jobs")
    gallery = importlib.import_module(MOD + ".gallery")
    ui = importlib.import_module(MOD + ".ui")
    draw_panels()


if args.real:

    @check("Z-Image-Turbo: промпт → картинка")
    def _():
        global image_path
        s.image_backend, s.size, s.seed = "zimage_turbo", "768x768", 42
        s.prompt = presets.PROMPTS["BERKUT"]
        job = run_job(bpy.ops.dronegen.generate_image, "txt2img", timeout=1200)
        assert job.status == "done", job.error
        image_path = gallery.selected(bpy.context)
        return f"{job.metrics.get('total_seconds')} с, пик VRAM {job.metrics.get('vram_peak_torch_gb')} ГБ"

    @check("TripoSR: картинка → меш в сцене")
    def _():
        s.source, s.mesh_backend, s.remove_bg, s.target_size = "GALLERY", "triposr", True, 2.0
        job = run_job(bpy.ops.dronegen.make_3d, "img2mesh", timeout=1200)
        assert job.status == "done", job.error
        obj = bpy.data.objects[job.output]
        check_mesh(obj, 2.0)
        return f"{len(obj.data.polygons)} полигонов, {job.metrics.get('total_seconds')} с, предупреждения: {job.warnings}"


Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
failed = [r["name"] for r in results if not r["ok"]]
if failed:
    raise RuntimeError("не прошли: " + ", ".join(failed))
