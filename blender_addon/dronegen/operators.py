"""Operators of the DroneGen panel. Each returns at once; the work runs in jobs.py threads."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import bpy
from bpy.props import IntProperty, StringProperty

from . import fx, gallery, importer, jobs, link, motion, presets, prefs, rig, safety, server_proc, studio, workspace
from .client import ServerError

MESH_TITLES = {"hunyuan": "Hunyuan3D", "hunyuan_mv": "Hunyuan3D-mv", "triposr": "TripoSR", "silhouette": "силуэт", "mock": "тест"}


def _fail(op, message: str):
    op.report({"ERROR"}, message)
    jobs.remember_error(message)
    return {"CANCELLED"}


def _client(op):
    """A configured client, or None after reporting why not."""
    try:
        client = prefs.client()
    except ServerError as e:
        _fail(op, str(e))
        return None
    if not client.local and not getattr(bpy.app, "online_access", True):
        _fail(op, "Сервер не на этом компьютере, а доступ в сеть у Blender выключен: "
                  "Edit → Preferences → System → Network → Allow Online Access")
        return None
    return client


class DRONEGEN_OT_check_server(bpy.types.Operator):
    """Спросить сервер, работает ли он, есть ли видеокарта и какие модели доступны"""

    bl_idname = "dronegen.check_server"
    bl_label = "Проверить соединение"

    def execute(self, context):
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        jobs.check_server(client)
        return {"FINISHED"}


class DRONEGEN_OT_start_server(bpy.types.Operator):
    """Запустить локальный сервер DroneGen в отдельном окне (из папки сервера в настройках аддона)"""

    bl_idname = "dronegen.start_server"
    bl_label = "Запустить сервер"

    def execute(self, context):
        p = prefs.get()
        try:
            message = server_proc.start(p.server_home, p.server_url)
        except RuntimeError as e:
            return _fail(self, str(e))
        self.report({"INFO"}, message)
        jobs.STATE.server.update(ok=None, message=message)
        client = _client(self)
        if client is not None:
            jobs.wait_for_server(client)
        return {"FINISHED"}


class DRONEGEN_OT_stop_server(bpy.types.Operator):
    """Остановить сервер (модели выгрузятся, видеопамять освободится)"""

    bl_idname = "dronegen.stop_server"
    bl_label = "Остановить сервер"

    def execute(self, context):
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        try:
            message = server_proc.stop(client)
        except RuntimeError as e:
            return _fail(self, str(e))
        jobs.STATE.server.update(ok=None, health=None, message=message)
        self.report({"INFO"}, message)
        return {"FINISHED"}


class DRONEGEN_OT_unload_models(bpy.types.Operator):
    """Выгрузить модели из памяти сервера: освободить видеопамять для Blender и других программ"""

    bl_idname = "dronegen.unload_models"
    bl_label = "Освободить видеопамять"

    def execute(self, context):
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        jobs.run_and_refresh(client, lambda c: c.unload(), "Не удалось выгрузить модели")
        self.report({"INFO"}, "Модели будут выгружены, как только сервер освободится")
        return {"FINISHED"}


class DRONEGEN_OT_generate_image(bpy.types.Operator):
    """Сгенерировать картинку по промпту на сервере. Готовая появится в галерее"""

    bl_idname = "dronegen.generate_image"
    bl_label = "Сгенерировать"

    def execute(self, context):
        s = context.scene.dronegen
        if not s.prompt.strip():
            return _fail(self, "Введите промпт (или выберите пресет)")
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        width, height = s.image_size()
        prompt = s.prompt.strip()
        params = {
            "prompt": prompt,
            "negative": s.negative if s.image_backend == "sdxl" else "",
            "width": width,
            "height": height,
            "steps": s.steps,
            "seed": s.seed,
            "backend": s.image_backend,
        }
        if s.style():
            params["style"] = s.style()  # the server adds the style words to the prompt
        meta = {"prompt": prompt, "prompt_user": prompt, "style": s.image_style, "backend": s.image_backend,
                "width": width, "height": height, "steps": s.steps, "subject": prompt}
        title = "Картинка: " + s.prompt.strip()[:32]
        jobs.start("txt2img", title, client, lambda c: c.txt2img(params), ("image",), gallery.folder(), meta)
        return {"FINISHED"}


@jobs.handler("txt2img")
def _image_done(job) -> str:
    info = dict(job.meta, seed=job.result.get("seed"), job=job.server_id, seconds=job.metrics.get("total_seconds"))
    info.pop("subject", None)
    info["prompt"] = job.result.get("prompt") or info["prompt"]  # with the style words the server added
    return gallery.add(Path(job.files["image"]), info)


def _object_name(path: Path) -> str:
    base = gallery.meta(path).get("prompt_user") or path.stem
    base = re.sub(r"\s+", " ", base).strip()[:40] or "figure"
    return f"DG {base}"


class DRONEGEN_OT_make_3d(bpy.types.Operator):
    """Сделать 3D-меш из выбранной картинки. Готовый меш появится в коллекции DroneGen в начале координат"""

    bl_idname = "dronegen.make_3d"
    bl_label = "Сделать 3D"

    def execute(self, context):
        s = context.scene.dronegen
        if s.source == "FILE":
            path = Path(bpy.path.abspath(s.image_file)) if s.image_file else None
            if path is None or not path.is_file():
                return _fail(self, "Выберите файл картинки")
        else:
            path = gallery.selected(context)
            if path is None:
                return _fail(self, "Сначала сгенерируйте или выберите картинку в галерее")
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        params = {
            "backend": s.mesh_backend,
            "remove_bg": s.remove_bg,
            "mc_resolution": int(s.mc_resolution),
            "target_faces": s.target_faces,
        }
        if s.mesh_backend == "silhouette":
            params["depth"] = s.depth
        if s.mesh_backend.startswith("hunyuan"):
            params["params"] = json.dumps({"symmetry": s.symmetry, "style": s.mesh_style})
        name = _object_name(path)
        meta = {"source": str(path), "backend": s.mesh_backend, "size": s.target_size, "name": name, "subject": name[3:]}
        title = f"3D ({MESH_TITLES.get(s.mesh_backend, s.mesh_backend)}): {path.stem[:24]}"
        folder = prefs.cache_dir() / "meshes"
        jobs.start("img2mesh", title, client, lambda c: c.img2mesh(path, params), ("mesh",), folder, meta)
        return {"FINISHED"}


@jobs.handler("img2mesh")
def _mesh_done(job) -> str:
    info = {"job": job.server_id, "backend": job.meta["backend"], "source": job.meta["source"]}
    obj = importer.import_glb(bpy.context, Path(job.files["mesh"]), job.meta["name"], job.meta["size"], info)
    return obj.name


class DRONEGEN_OT_cancel_job(bpy.types.Operator):
    """Отменить задачу"""

    bl_idname = "dronegen.cancel_job"
    bl_label = "Отменить"

    key: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        job = jobs.STATE.jobs.get(self.key)
        if job is not None and job.active:
            job.cancel_requested = True
            job.stage = "отмена…"
        return {"FINISHED"}


class DRONEGEN_OT_forget_job(bpy.types.Operator):
    """Убрать строку о завершённой задаче"""

    bl_idname = "dronegen.forget_job"
    bl_label = "Убрать"

    key: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        jobs.forget(self.key)
        jobs.redraw()
        return {"FINISHED"}


class DRONEGEN_OT_dismiss(bpy.types.Operator):
    """Скрыть сообщения об ошибках"""

    bl_idname = "dronegen.dismiss"
    bl_label = "Скрыть"

    def execute(self, context):
        jobs.STATE.errors.clear()
        for key in [k for k, j in jobs.STATE.jobs.items() if j.status in ("error", "cancelled")]:
            del jobs.STATE.jobs[key]
        return {"FINISHED"}


def show_image(context, image) -> str:
    """Show an image in an Image Editor: an open one, or a new small window."""
    screen = context.screen
    if screen is not None:
        for area in screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = image
                return "в открытом Image Editor"
    if bpy.app.background or context.window is None or not context.window_manager.windows:
        return "загружена в Blender (окна нет)"
    before = {w.as_pointer() for w in context.window_manager.windows}
    try:
        bpy.ops.screen.userpref_show("INVOKE_DEFAULT")  # a new small window, turned into an Image Editor
    except RuntimeError:
        return "загружена: откройте её в любом Image Editor"
    new = [w for w in context.window_manager.windows if w.as_pointer() not in before]
    if not new:
        return "загружена: откройте её в любом Image Editor"
    area = new[0].screen.areas[0]
    area.type = "IMAGE_EDITOR"
    area.spaces.active.image = image
    return "в новом окне"


class DRONEGEN_OT_open_image_editor(bpy.types.Operator):
    """Открыть выбранную картинку в Image Editor"""

    bl_idname = "dronegen.open_image_editor"
    bl_label = "Открыть в Image Editor"

    @classmethod
    def poll(cls, context):
        return gallery.selected(context) is not None

    def execute(self, context):
        path = gallery.selected(context)
        image = bpy.data.images.load(str(path), check_existing=True)
        where = show_image(context, image)
        self.report({"INFO"}, f"{image.name}: {where}")
        return {"FINISHED"}


class DRONEGEN_OT_open_folder(bpy.types.Operator):
    """Открыть папку с картинками и мешами"""

    bl_idname = "dronegen.open_folder"
    bl_label = "Открыть папку"

    def execute(self, context):
        folder = str(gallery.folder())
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606 - opens Explorer
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
        return {"FINISHED"}


class DRONEGEN_OT_refresh_gallery(bpy.types.Operator):
    """Перечитать папку с картинками"""

    bl_idname = "dronegen.refresh_gallery"
    bl_label = "Обновить галерею"

    def execute(self, context):
        gallery.refresh()
        return {"FINISHED"}


class DRONEGEN_OT_gallery_step(bpy.types.Operator):
    """Соседняя картинка галереи"""

    bl_idname = "dronegen.gallery_step"
    bl_label = "Листать галерею"

    delta: IntProperty(default=1, options={"HIDDEN"})

    def execute(self, context):
        gallery.step(self.delta)
        jobs.redraw()
        return {"FINISHED"}


class DRONEGEN_OT_use_preset(bpy.types.Operator):
    bl_idname = "dronegen.use_preset"
    bl_label = "Готовый промпт"
    bl_options = {"UNDO"}

    key: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        return presets.PROMPTS.get(properties.key, "Подставить готовый промпт")

    def execute(self, context):
        if self.key not in presets.PROMPTS:
            return {"CANCELLED"}
        context.scene.dronegen.prompt = presets.PROMPTS[self.key]
        return {"FINISHED"}


class DRONEGEN_OT_select_object(bpy.types.Operator):
    """Выделить модель и показать её во вьюпорте"""

    bl_idname = "dronegen.select_object"
    bl_label = "Показать модель"

    name: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        obj = context.view_layer.objects.get(self.name)
        if obj is None:
            self.report({"WARNING"}, f"«{self.name}» больше нет в сцене")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            self.report({"WARNING"}, "Выделить модель можно в Object Mode")
            return {"CANCELLED"}
        for o in context.view_layer.objects:
            if o.select_get():
                o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        link.frame_selected(context)
        return {"FINISHED"}


class DRONEGEN_OT_delete_image(bpy.types.Operator):
    """Удалить картинку из галереи (файл в папке кэша)"""

    bl_idname = "dronegen.delete_image"
    bl_label = "Удалить картинку"

    name: StringProperty(options={"HIDDEN"})

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        gallery.remove(self.name)
        return {"FINISHED"}


class DRONEGEN_OT_open_studio(bpy.types.Operator):
    """Открыть студию DroneGen: окно с чатом, проектами и 3D-просмотром. Сервер запустится сам, если он выключен"""

    bl_idname = "dronegen.open_studio"
    bl_label = "Открыть студию"

    def execute(self, context):
        error = studio.open_async()
        if error:
            return _fail(self, error)
        self.report({"INFO"}, "Открываю студию DroneGen…")
        return {"FINISHED"}


class DRONEGEN_OT_open_workspace(bpy.types.Operator):
    """Перейти в воркспейс DroneGen: картинки слева, 3D справа, структура и свойства (сцена не меняется)"""

    bl_idname = "dronegen.open_workspace"
    bl_label = "Воркспейс DroneGen"

    def execute(self, context):
        try:
            message = workspace.open_workspace(context)
        except Exception as e:  # noqa: BLE001
            return _fail(self, f"Воркспейс DroneGen: {e}")
        if message:
            self.report({"INFO"}, message)
        return {"FINISHED"}


def _figure(op, context):
    """The figure mesh behind the active object (the mesh, its drone points or its rig), or None after a report."""
    mesh = rig.figure_of(context.active_object)
    if mesh is None:
        _fail(op, "Выберите модель: меш DroneGen, его точки дронов или его скелет")
    return mesh


class DRONEGEN_OT_build_rig(bpy.types.Operator):
    """Построить скелет для выбранной модели: кости ног, крыльев, рук, шеи и хвоста по её форме.
    Если «Сразу привязать», меш и точки дронов сразу начнут двигаться вместе с костями"""

    bl_idname = "dronegen.build_rig"
    bl_label = "Построить скелет"

    def execute(self, context):
        mesh = _figure(self, context)
        if mesh is None:
            return {"CANCELLED"}
        if not len(mesh.data.polygons):
            return _fail(self, f"У «{mesh.name}» нет граней: скелет строится по объёму модели")
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        s = context.scene.dronegen
        data = rig.mesh_npz(mesh)
        params = {"plan": s.rig_plan, "method": s.rig_method}
        meta = {"object": mesh.name, "bind": s.rig_bind, "subject": f"Скелет: {mesh.name}"}
        folder = prefs.cache_dir() / "rigs"
        jobs.start("rig", f"Скелет: {mesh.name[:28]}", client, lambda c: c.rig(data, params), ("rig",), folder, meta)
        return {"FINISHED"}


@jobs.handler("rig")
def _rig_done(job) -> str:
    mesh = bpy.data.objects.get(job.meta["object"])
    if mesh is None:
        raise RuntimeError(f"модели «{job.meta['object']}» больше нет в сцене")
    armature = rig.build(bpy.context, mesh, job.result)
    text = "Скелет готов"  # what was found is shown under it, in the rig's box
    if job.meta.get("bind"):
        stats = rig.bind(bpy.context, mesh, armature)
        text += " и привязан" + (" с точками дронов" if len(stats["objects"]) > 1 else "")
    job.warnings = list(job.warnings) + list(job.result.get("notes") or [])
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    return text


class DRONEGEN_OT_bind_rig(bpy.types.Operator):
    """Привязать модель и её точки дронов к скелету заново: после правки костей или новых точек Pointille"""

    bl_idname = "dronegen.bind_rig"
    bl_label = "Привязать"

    def execute(self, context):
        mesh = _figure(self, context)
        if mesh is None:
            return {"CANCELLED"}
        armature = rig.rig_of(mesh)
        if armature is None:
            return _fail(self, "У модели ещё нет скелета: «Построить скелет»")
        try:
            stats = rig.bind(context, mesh, armature)
        except Exception as e:  # noqa: BLE001
            return _fail(self, f"Не удалось привязать: {e}")
        points = len(stats["objects"]) - 1
        self.report({"INFO"}, f"Привязано за {stats['seconds']} с" + (" вместе с точками дронов" if points else ""))
        return {"FINISHED"}


class DRONEGEN_OT_rig_mode(bpy.types.Operator):
    """Править кости (режим правки, левая и правая стороны зеркально) или пробовать позы (режим позы)"""

    bl_idname = "dronegen.rig_mode"
    bl_label = "Режим скелета"

    mode: bpy.props.EnumProperty(items=[("EDIT", "Править", ""), ("POSE", "Поза", ""), ("OBJECT", "Готово", "")],
                                 options={"HIDDEN"})

    def execute(self, context):
        mesh = _figure(self, context)
        armature = rig.rig_of(mesh) if mesh is not None else None
        if armature is None:
            return _fail(self, "У модели ещё нет скелета: «Построить скелет»")
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if self.mode == "OBJECT":
            return {"FINISHED"}
        for o in context.view_layer.objects:
            o.select_set(False)
        armature.hide_set(False)
        armature.select_set(True)
        context.view_layer.objects.active = armature
        armature.data.use_mirror_x = True  # the bones lie in the object's space: X mirrors left and right
        armature.pose.use_mirror_x = True
        bpy.ops.object.mode_set(mode=self.mode)
        if self.mode == "EDIT":
            self.report({"INFO"}, "Двигайте суставы; потом «Готово» и «Привязать», чтобы веса пересчитались")
        return {"FINISHED"}


class DRONEGEN_OT_reset_pose(bpy.types.Operator):
    """Вернуть все кости в исходную позу (позы, сделанные вручную, сбросятся; анимация останется)"""

    bl_idname = "dronegen.reset_pose"
    bl_label = "Сбросить позу"

    def execute(self, context):
        mesh = _figure(self, context)
        armature = rig.rig_of(mesh) if mesh is not None else None
        if armature is None:
            return {"CANCELLED"}
        for pb in armature.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.scale = (1, 1, 1)
        return {"FINISHED"}


class DRONEGEN_OT_remove_rig(bpy.types.Operator):
    """Удалить скелет модели: меш и точки дронов вернутся в исходную форму"""

    bl_idname = "dronegen.remove_rig"
    bl_label = "Удалить скелет"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        mesh = _figure(self, context)
        armature = rig.rig_of(mesh) if mesh is not None else None
        if armature is None:
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        name = armature.name
        rig.remove(mesh, armature)
        self.report({"INFO"}, f"Скелет «{name}» удалён")
        return {"FINISHED"}


# -- animation layers ----------------------------------------------------------------------------
def _rig(op, context):
    """The skeleton of the selected figure, or None after a report."""
    mesh = _figure(op, context)
    armature = rig.rig_of(mesh) if mesh is not None else None
    if mesh is not None and armature is None:
        _fail(op, "У модели ещё нет скелета: «Построить скелет»")
    return armature


def _catalog(op) -> bool:
    """The server's motion catalog, fetched once (a quick local request)."""
    if motion.CATALOG:
        return True
    client = _client(op)
    if client is None:
        return False
    try:
        motion.fetch(client)
    except ServerError as e:
        _fail(op, f"Каталог движений: {e}")
        return False
    return True


def _defaults(layer, context, armature) -> None:
    """Frames, period and fades that suit the layer's motion (or flight)."""
    rate = motion.fps(context.scene)
    layer.start = context.scene.frame_current
    scale = max(armature.matrix_world.to_scale())  # sizes of layers are metres in the sky
    size = float(rig.info(armature).get("size") or 0.0) * scale or max(armature.dimensions)
    if layer.kind == "PATH":
        layer.name, layer.radius, layer.laps = "Полёт", round(2.5 * size, 1), 1.0
        layer.end, layer.fade_in, layer.fade_out = layer.start + int(30 * rate), 2.0, 2.0
        return
    if layer.kind == "SHAPE":
        layer.name = layer.bl_rna.properties["shape_fx"].enum_items[layer.shape_fx].name
        layer.period = 3.0
        layer.fx_size = {"PULSE": 0.5, "SCATTER": round(0.5 * size, 2)}.get(layer.shape_fx, round(0.05 * size, 2))
        layer.end, layer.fade_in, layer.fade_out = layer.start + int(10 * rate), 1.0, 1.0
        return
    if layer.kind == "EFFECT":
        layer.name = layer.bl_rna.properties["effect"].enum_items[layer.effect].name
        layer.size, layer.travel = round(max(0.6 * size, fx.min_size(layer.effect, context.scene)), 1), 3.0
        layer.end, layer.fade_in, layer.fade_out = layer.start + int(5 * rate), 0.0, 1.0
        if not layer.bone:  # a trail and a part's place need a part: what is held, an arm, a wing, the head
            chains = {b.get("dg_chain"): b.get("dg_role") for b in armature.data.bones}
            for role in ("prop", "arm", "wing", "head", "leg_front", "tail"):
                layer.bone = next((c for c, r in sorted(chains.items()) if r == role and c), layer.bone)
                if layer.bone:
                    break
        return
    m = motion.CATALOG.get(layer.motion) or {}
    layer.name = m.get("title", "Движение")
    layer.period = float(m.get("period") or 4.0)
    kind = m.get("kind", "cycle")
    if kind == "pose":
        layer.end, layer.fade_in, layer.fade_out = layer.start + int(4 * rate), 1.0, 1.0
    elif kind == "once":
        layer.end, layer.fade_in, layer.fade_out = layer.start + int(layer.period * rate), 0.2, 0.2
    else:
        layer.end = layer.start + int(max(10.0, 2 * layer.period) * rate)
        layer.fade_in = layer.fade_out = min(1.0, layer.period / 2)


class DRONEGEN_OT_layer_add(bpy.types.Operator):
    """Добавить слой анимации: движение частей тела или полёт по пути"""

    bl_idname = "dronegen.layer_add"
    bl_label = "Добавить слой"

    kind: bpy.props.EnumProperty(items=[("MOTION", "Движение", ""), ("PATH", "Полёт", ""), ("SHAPE", "Эффект формы", ""),
                                        ("EFFECT", "Эффект с дронами", "")], options={"HIDDEN"})
    motion: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None or (self.kind == "MOTION" and not _catalog(self)):
            return {"CANCELLED"}
        layer = armature.dronegen_layers.add()
        layer.kind = self.kind
        if self.kind == "MOTION":
            fitting = [k for k, _m in motion.fitting(armature)]
            layer.motion = self.motion if self.motion in fitting else (fitting[0] if fitting else "")
        _defaults(layer, context, armature)
        armature.dronegen_layer = len(armature.dronegen_layers) - 1
        return {"FINISHED"}


class DRONEGEN_OT_layer_remove(bpy.types.Operator):
    """Убрать выбранный слой (дорожка исчезнет после «Применить»)"""

    bl_idname = "dronegen.layer_remove"
    bl_label = "Убрать слой"

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None or not armature.dronegen_layers:
            return {"CANCELLED"}
        i = armature.dronegen_layer
        armature.dronegen_layers.remove(i)
        armature.dronegen_layer = max(0, min(i, len(armature.dronegen_layers) - 1))
        return {"FINISHED"}


class DRONEGEN_OT_layer_move(bpy.types.Operator):
    """Сдвинуть слой выше или ниже в списке"""

    bl_idname = "dronegen.layer_move"
    bl_label = "Сдвинуть слой"

    step: IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None:
            return {"CANCELLED"}
        i, n = armature.dronegen_layer, len(armature.dronegen_layers)
        j = i + self.step
        if 0 <= i < n and 0 <= j < n:
            armature.dronegen_layers.move(i, j)
            armature.dronegen_layer = j
        return {"FINISHED"}


class DRONEGEN_OT_set_motion(bpy.types.Operator):
    """Движение выбранного слоя"""

    bl_idname = "dronegen.set_motion"
    bl_label = "Движение"

    key: StringProperty(options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        return (motion.CATALOG.get(properties.key) or {}).get("about", "")

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None or not armature.dronegen_layers:
            return {"CANCELLED"}
        layer = armature.dronegen_layers[armature.dronegen_layer]
        layer.kind, layer.motion = "MOTION", self.key
        start = layer.start
        _defaults(layer, context, armature)
        layer.end += start - layer.start  # keep where the layer was
        layer.start = start
        return {"FINISHED"}


class DRONEGEN_OT_fetch_motions(bpy.types.Operator):
    """Обновить список движений с сервера"""

    bl_idname = "dronegen.fetch_motions"
    bl_label = "Обновить движения"

    def execute(self, context):
        motion.CATALOG.clear()
        return {"FINISHED"} if _catalog(self) else {"CANCELLED"}


class DRONEGEN_OT_apply_layers(bpy.types.Operator):
    """Собрать анимацию из слоёв: сервер рассчитывает движения для этого скелета, слои становятся дорожками NLA.
    Меш и точки дронов двигаются вместе со скелетом"""

    bl_idname = "dronegen.apply_layers"
    bl_label = "Применить"

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None or not _catalog(self):
            return {"CANCELLED"}
        missing = [layer.name for layer in armature.dronegen_layers
                   if layer.enabled and layer.kind == "MOTION" and layer.motion not in motion.CATALOG]
        if missing:
            return _fail(self, f"Не выбрано движение у слоёв: {', '.join(missing)}")
        client = _client(self)
        if client is None:
            return {"CANCELLED"}
        try:
            effects = motion.effect_requests(context, armature)
        except RuntimeError as e:
            return _fail(self, str(e))
        data, todo = motion.skeleton(armature), motion.requests(armature)
        jobs.run("anim", f"Анимация: {armature.name[:28]}", client, lambda c: motion.bake_all(c, data, todo, effects),
                 {"rig": armature.name, "effects": effects, "subject": f"Анимация: {armature.name}"})
        return {"FINISHED"}


@jobs.handler("anim")
def _anim_done(job) -> str:
    armature = bpy.data.objects.get(job.meta["rig"])
    if armature is None:
        raise RuntimeError(f"скелета «{job.meta['rig']}» больше нет в сцене")
    made = motion.build(bpy.context, armature, job.result, job.meta.get("effects"))
    job.warnings = list(job.warnings) + made.get("notes", [])
    trails = motion.trails(bpy.context, armature)  # a trail follows the tip of the animation just built
    if trails:
        client = prefs.client()
        jobs.run("trail", f"След: {armature.name[:28]}", client,
                 lambda c: {i: c.effect(motion._public(req)) for i, req in trails.items()},
                 {"rig": armature.name, "requests": trails, "subject": f"След: {armature.name}"})
    words = (("motions", "движений"), ("paths", "полётов"), ("shapes", "эффектов формы"), ("effects", "эффектов с дронами"))
    parts = [f"{word}: {made[key]}" for key, word in words if made.get(key)] or ["пусто"]
    return "Анимация собрана · " + ", ".join(parts)


@jobs.handler("trail")
def _trail_done(job) -> str:
    armature = bpy.data.objects.get(job.meta["rig"])
    if armature is None:
        raise RuntimeError(f"скелета «{job.meta['rig']}» больше нет в сцене")
    notes: list = []
    made = motion.place_effects(bpy.context, armature, job.meta["requests"], job.result, notes)
    job.warnings = list(job.warnings) + notes
    return f"След движения: эффектов {made}"


class DRONEGEN_OT_clear_layers(bpy.types.Operator):
    """Убрать собранную анимацию (дорожки, действия и пути DroneGen); слои в списке остаются"""

    bl_idname = "dronegen.clear_layers"
    bl_label = "Убрать анимацию"

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None:
            return {"CANCELLED"}
        motion.clear(armature)
        return {"FINISHED"}


class DRONEGEN_OT_check_drones(bpy.types.Operator):
    """Проверить, смогут ли настоящие дроны это пролететь: скорость, подъём, спуск, ускорение и расстояние между
    точками дронов на всём протяжении анимации (пороги — из настроек безопасности Skybrush)"""

    bl_idname = "dronegen.check_drones"
    bl_label = "Проверить дронов"

    def execute(self, context):
        mesh = rig.figure_of(context.active_object)
        armature = rig.rig_of(mesh) if mesh is not None else None
        try:
            report = safety.check(context, armature)
        except RuntimeError as e:
            return _fail(self, str(e))
        if report["ok"]:
            self.report({"INFO"}, f"Дроны справятся: {report['drones']} точек, кадры {report['frames'][0]}–{report['frames'][1]}")
        else:
            self.report({"WARNING"}, report["problems"][0])
        return {"FINISHED"}


class DRONEGEN_OT_slow_down(bpy.types.Operator):
    """Замедлить все слои (периоды и плавные переходы длиннее, кругов полёта меньше) и собрать анимацию заново"""

    bl_idname = "dronegen.slow_down"
    bl_label = "Замедлить"

    factor: bpy.props.FloatProperty(default=1.5, min=1.0, options={"HIDDEN"})

    def execute(self, context):
        armature = _rig(self, context)
        if armature is None:
            return {"CANCELLED"}
        safety.slow_down(armature, self.factor)
        armature.pop(safety.REPORT_KEY, None)  # the old report no longer tells the truth
        return bpy.ops.dronegen.apply_layers()


classes = (
    DRONEGEN_OT_open_studio,
    DRONEGEN_OT_open_workspace,
    DRONEGEN_OT_check_server,
    DRONEGEN_OT_start_server,
    DRONEGEN_OT_stop_server,
    DRONEGEN_OT_unload_models,
    DRONEGEN_OT_generate_image,
    DRONEGEN_OT_make_3d,
    DRONEGEN_OT_cancel_job,
    DRONEGEN_OT_forget_job,
    DRONEGEN_OT_dismiss,
    DRONEGEN_OT_open_image_editor,
    DRONEGEN_OT_open_folder,
    DRONEGEN_OT_refresh_gallery,
    DRONEGEN_OT_gallery_step,
    DRONEGEN_OT_use_preset,
    DRONEGEN_OT_select_object,
    DRONEGEN_OT_delete_image,
    DRONEGEN_OT_build_rig,
    DRONEGEN_OT_bind_rig,
    DRONEGEN_OT_rig_mode,
    DRONEGEN_OT_reset_pose,
    DRONEGEN_OT_remove_rig,
    DRONEGEN_OT_layer_add,
    DRONEGEN_OT_layer_remove,
    DRONEGEN_OT_layer_move,
    DRONEGEN_OT_set_motion,
    DRONEGEN_OT_fetch_motions,
    DRONEGEN_OT_apply_layers,
    DRONEGEN_OT_clear_layers,
    DRONEGEN_OT_check_drones,
    DRONEGEN_OT_slow_down,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
