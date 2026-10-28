"""Runs inside a Blender WITH a window (see run_blender_e2e.py --gui): the parts a background run cannot see.

A model sent from the Studio arrives in the scene, the DroneGen workspace becomes the window's
workspace, the hotkey is registered, a generated creature gets a skeleton and a pose. Screenshots of the whole window are saved for a look.
Steps run from a timer, because the window only updates while Blender's event loop runs.
"""

import argparse
import importlib
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import addon_utils
import bpy

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args(argv)

MOD = "bl_ext.user_default.dronegen"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
results = []


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(args.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def shot(name):
    win = bpy.context.window_manager.windows[0]
    path = str(OUT / f"{name}.png")
    with bpy.context.temp_override(window=win, screen=win.screen):
        bpy.ops.screen.screenshot(filepath=path, check_existing=False)
    return path


def record(name, ok, info=""):
    results.append({"name": name, "ok": ok, "info": str(info)})
    print(("OK   " if ok else "FAIL ") + name, info, flush=True)


if MOD not in bpy.context.preferences.addons:
    addon_utils.enable(MOD, default_set=True)
prefs = bpy.context.preferences.addons[MOD].preferences
prefs.server_url, prefs.cache_dir = args.url, args.cache
link = importlib.import_module(MOD + ".link")
workspace = importlib.import_module(MOD + ".workspace")
studio = importlib.import_module(MOD + ".studio")
jobs = importlib.import_module(MOD + ".jobs")
CREATURE = "a1a8689d795c"  # a generated white pegasus in runtime/outputs, if this computer has it

stage = {"n": 0, "t": time.time(), "item": None}


def finish():
    (OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    bpy.ops.wm.quit_blender()


def tick():
    try:
        return step()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        record(f"stage {stage['n']}", False, f"{type(e).__name__}: {e}")
        finish()
        return None


def step():
    n, waited = stage["n"], time.time() - stage["t"]

    def go(next_n):
        stage.update(n=next_n, t=time.time())
        return 0.5

    if n == 0:  # a mesh from the Studio inbox
        project = api("POST", "/studio/api/projects", {"title": "GUI"})
        chat = api("POST", "/studio/api/chats", {"project_id": project["id"]})
        # a picture and a mock mesh through the server's own queue
        img = api("POST", "/txt2img", {"prompt": "gui smoke", "backend": "mock", "steps": 2, "width": 512, "height": 512})
        stage["job"] = img["job_id"]
        stage["chat"] = chat["id"]
        return go(1)
    if n == 1:
        job = api("GET", f"/jobs/{stage['job']}")
        if job["status"] != "done":
            return 0.3 if waited < 60 else go(99)
        stage["image"] = job["files"]["image"]
        boundary = "x"
        with open(OUT / "pic.png", "wb") as f:
            f.write(urllib.request.urlopen(args.url + f"/files/{stage['image']}", timeout=30).read())
        req = urllib.request.Request(args.url + "/img2mesh", method="POST", data=(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"backend\"\r\n\r\nsilhouette\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"mc_resolution\"\r\n\r\n96\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image_name\"\r\n\r\n{stage['image']}\r\n--{boundary}--\r\n"
        ).encode(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        stage["mesh_job"] = json.loads(urllib.request.urlopen(req, timeout=30).read())["job_id"]
        return go(2)
    if n == 2:
        job = api("GET", f"/jobs/{stage['mesh_job']}")
        if job["status"] != "done":
            return 0.3 if waited < 90 else go(99)
        stage["item"] = api("POST", f"/studio/api/chats/{stage['chat']}/blender",
                            {"file": job["files"]["mesh"], "name": "GUI смоук"})
        link._state["next"] = 0.0
        return go(3)
    if n == 3:  # the add-on's own timer takes the model
        names = [o.name for o in bpy.data.objects if o.name.startswith("DG GUI смоук")]
        if not names:
            return 0.5 if waited < 40 else go(99)
        record("модель из студии пришла в сцену", True, names[0])
        return go(4)
    if n == 4:
        shot("1_scene_after_import")
        kc = bpy.context.window_manager.keyconfigs.addon
        items = [(km.name, kmi.type, kmi.alt, kmi.shift) for km in kc.keymaps for kmi in km.keymap_items
                 if kmi.idname == "dronegen.open_studio"]
        record("горячая клавиша Alt+Shift+D зарегистрирована", items == [("Window", "D", True, True)], items)
        opened = []
        stage["real_open"] = studio.open_window
        studio.open_window = lambda url, profile: opened.append(url) or "тест"
        bpy.ops.dronegen.open_studio()
        stage["opened"] = opened
        return go(5)
    if n == 5:
        if not stage["opened"]:
            return 0.5 if waited < 15 else go(99)
        studio.open_window = stage["real_open"]
        record("оператор «Открыть студию» открывает окно", stage["opened"][0] == args.url + "/studio/", stage["opened"])
        bpy.ops.dronegen.open_workspace()
        return go(6)
    if n == 6:
        win = bpy.context.window_manager.windows[0]
        if win.workspace.name != "DroneGen":
            return 0.5 if waited < 60 else go(99)
        return go(7)
    if n == 7:
        win = bpy.context.window_manager.windows[0]
        kinds = sorted(a.type for a in win.screen.areas)
        record("воркспейс DroneGen открыт", {"IMAGE_EDITOR", "VIEW_3D", "OUTLINER", "PROPERTIES"} <= set(kinds), kinds)
        obj = bpy.context.view_layer.objects.active
        record("активный объект и режим не изменились", obj is not None and obj.name.startswith("DG GUI") and obj.mode == "OBJECT",
               (obj.name, obj.mode) if obj else None)
        return go(8)
    if n == 8:
        shot("2_workspace")
        # the Image Editor shows the picture of the figure
        win = bpy.context.window_manager.windows[0]
        img = bpy.data.images.load(str(OUT / "pic.png"), check_existing=True)
        for a in win.screen.areas:
            if a.type == "IMAGE_EDITOR":
                a.spaces.active.image = img
        return go(9)
    if n == 9:
        shot("3_workspace_with_picture")
        # a real generated creature (the white pegasus of runtime/outputs) gets a skeleton
        src = Path(__file__).resolve().parents[1] / "runtime" / "outputs" / f"{CREATURE}_mesh.glb"
        if not src.is_file():
            record("скелет на настоящей модели", True, f"пропущено: нет {src.name}")
            finish()
            return None
        dest = Path(args.cache).parent / "server" / "outputs" / src.name
        dest.write_bytes(src.read_bytes())
        api("POST", f"/studio/api/chats/{stage['chat']}/blender", {"file": src.name, "name": "GUI тулпар"})
        link._state["next"] = 0.0
        return go(10)
    if n == 10:
        horse = next((o for o in bpy.data.objects if o.name.startswith("DG GUI тулпар")), None)
        if horse is None:
            return 0.5 if waited < 40 else go(99)
        # drone points on it, as Pointille's «Распределить вершины» would make them: they get bound with the mesh
        step = max(1, len(horse.data.vertices) // 300)
        me = bpy.data.meshes.new("GUI points")
        me.from_pydata([tuple(v.co) for v in list(horse.data.vertices)[::step]], [], [])
        points = bpy.data.objects.new(horse.name + "_points", me)
        bpy.context.scene.collection.objects.link(points)
        points.parent, points["pointille_drones"] = horse, 1
        for o in bpy.context.view_layer.objects:
            o.select_set(o is horse)
        bpy.context.view_layer.objects.active = horse
        s = bpy.context.scene.dronegen
        s.rig_plan, s.rig_method, s.rig_bind = "auto", "template", True
        stage["rig_before"] = {j.key for j in jobs.jobs_of("rig")}
        bpy.ops.dronegen.build_rig()
        return go(11)
    if n == 11:
        job = next((j for j in jobs.jobs_of("rig") if j.key not in stage["rig_before"]), None)
        if job is None or job.active:
            return 0.5 if waited < 120 else go(99)
        record("скелет построен и привязан", job.status == "done", job.output or job.error)
        armature = bpy.context.view_layer.objects.active
        horse = next(o for o in bpy.data.objects if o.name.startswith("DG GUI тулпар") and o.type == "MESH")
        for o in bpy.context.view_layer.objects:  # only the creature, its rig and its points: no startup cube around it
            if o not in (horse, armature) and o.parent is not horse:
                o.hide_set(True)
        bpy.context.scene.frame_current = 1
        for kind, key in (("MOTION", "wing_flap"), ("MOTION", "gallop"), ("PATH", ""), ("SHAPE", "")):  # all at once
            bpy.ops.dronegen.layer_add(kind=kind, motion=key)
        bpy.context.scene.frame_current = 50  # a star over the figure, with drones of its own, from frame 50
        bpy.ops.dronegen.layer_add(kind="EFFECT")
        bpy.context.scene.frame_current = 1
        stage["anim_before"] = {j.key for j in jobs.jobs_of("anim")}
        bpy.ops.dronegen.apply_layers()
        return go(12)
    if n == 12:
        job = next((j for j in jobs.jobs_of("anim") if j.key not in stage["anim_before"]), None)
        if job is None or job.active:
            return 0.5 if waited < 60 else go(99)
        record("слои анимации собраны", job.status == "done", job.output or job.error)
        fx = importlib.import_module(MOD + ".fx")
        made = fx.effects_of(bpy.data.objects[job.meta["rig"]])
        record("звезда со своими дронами", len(made) == 1 and len(made[0].data.vertices) > 3,
               f"{[(o.name, len(o.data.vertices)) for o in made]}; {job.warnings[:2]}")
        bpy.ops.dronegen.check_drones()
        bpy.context.scene.frame_set(70)  # out on the circle, wings and legs moving
        win = bpy.context.window_manager.windows[0]
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.show_region_ui = True
                window_region = next(r for r in area.regions if r.type == "WINDOW")
                with bpy.context.temp_override(window=win, area=area, region=window_region):
                    bpy.ops.view3d.view_all()
                ui_region = next(r for r in area.regions if r.type == "UI")
                try:
                    ui_region.active_panel_category = "DroneGen"
                except (AttributeError, TypeError):  # read-only before Blender 4.2
                    pass
        return go(13)
    if n == 13:  # the sidebar is laid out now: scroll down to the animation panels at its bottom
        win = bpy.context.window_manager.windows[0]
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                ui_region = next(r for r in area.regions if r.type == "UI")
                with bpy.context.temp_override(window=win, area=area, region=ui_region):
                    for _ in range(80):
                        bpy.ops.view2d.scroll_down()
        return go(14)
    if n == 14:
        shot("4_animation")
        finish()
        return None
    # n == 99: a wait timed out
    record(f"таймаут на этапе {n}", False, f"прошло {waited:.0f} с")
    finish()
    return None


bpy.app.timers.register(tick, first_interval=3.0)
