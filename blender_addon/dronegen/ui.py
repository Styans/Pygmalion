"""N-panel «DroneGen» in the 3D Viewport: the Studio and the server on top, then picture, gallery, 3D and animation.

Blender labels never wrap and cut long text with «…», so messages are wrapped here by their
real width in the UI font (``blf``). Rarely used settings sit in folds that start closed.
"""

from __future__ import annotations

import math
import textwrap
import urllib.parse

import blf
import bpy

from . import fx, gallery, jobs, link, motion, presets, prefs, rig, safety, studio

# Icons get renamed between Blender versions: the first name this Blender has wins.
DOT_OK = ("STRIP_COLOR_04", "SEQUENCE_COLOR_04", "CHECKMARK")
DOT_ERROR = ("STRIP_COLOR_01", "SEQUENCE_COLOR_01", "ERROR")
DOT_WAIT = ("STRIP_COLOR_03", "SEQUENCE_COLOR_03", "SORTTIME")
KIND_ICONS = {"txt2img": "IMAGE_DATA", "img2mesh": "MESH_MONKEY", "rig": "ARMATURE_DATA", "anim": "NLA", "trail": "CURVE_PATH"}

STYLE_BUTTONS = (  # the style switch: (style, text on the button, icon)
    ("render", "Объёмный", "SHADING_RENDERED"),
    ("sheet", "4 вида", "IMGDISPLAY"),
    ("flat", "Плоский", "IMAGE_ALPHA"),
    ("none", "Без стиля", "TEXT"),
)
STYLE_HINTS = {
    "render": "Затенённый 3D-рендер → Hunyuan3D-mini или TripoSR",
    "sheet": "Спереди, слева, сзади и справа на одном листе → Hunyuan3D-mv, самая точная форма",
    "flat": "Чёткий плоский силуэт → «Силуэт → объём»",
    "none": "Промпт уходит как есть, без слов стиля",
}
STYLE_NAMES = {"render": "объёмный", "sheet": "лист 4 вида", "flat": "плоский"}
IMAGE_BACKEND_NAMES = {"zimage_turbo": "Z-Image", "sdxl": "SDXL", "mock": "тест"}

# Room that the panel takes from the region width before text starts (pixels at UI scale 1):
# category tabs, panel margins, the label's own padding; a box and an icon take more.
TEXT_MARGIN = 54
BOX_MARGIN = 14
ICON_WIDTH = 22

_icons: set | None = None
_PROP_ARGS = set(bpy.types.UILayout.bl_rna.functions["prop"].parameters.keys())


def icon(*names: str) -> str:
    global _icons
    if _icons is None:
        _icons = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
    return next((n for n in names if n in _icons), "NONE")


# -- text ------------------------------------------------------------------------------------
def _prefs(context):
    return getattr(context, "preferences", None) or bpy.context.preferences


def _ui_scale(context) -> float:
    """The interface scale; 0 in a Blender without a window that never drew, taken as 1."""
    return _prefs(context).system.ui_scale or 1.0


def text_lines(text: str, width: float, context) -> list[str]:
    """``text`` split into lines no wider than ``width`` pixels in the UI font."""
    p = _prefs(context)
    try:
        blf.size(0, p.ui_styles[0].widget.points * _ui_scale(context))
        measure = lambda s: blf.dimensions(0, s)[0]  # noqa: E731
        measure("Ж")
    except Exception:  # noqa: BLE001 - no font (e.g. a test without UI): estimate by characters
        return textwrap.wrap(text, max(12, int(width / 7.0))) or [""]
    lines = []
    for para in text.splitlines() or [""]:
        line = ""
        for word in para.split():
            while measure(word) > width and len(word) > 8:  # a long address: cut it by characters
                cut = len(word) - 1
                while cut > 1 and measure(word[:cut]) > width:
                    cut -= 1
                if line:
                    lines.append(line)
                    line = ""
                lines.append(word[:cut])
                word = word[cut:]
            candidate = f"{line} {word}" if line else word
            if line and measure(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def wrap(layout, text: str, context, icon: str = "NONE", alert: bool = False, muted: bool = False,
         boxed: bool = False) -> None:
    """A label that wraps instead of being cut. ``boxed``: it sits inside ``layout.box()``."""
    region = getattr(context, "region", None)
    scale = _ui_scale(context)
    margin = TEXT_MARGIN + (BOX_MARGIN if boxed else 0) + (ICON_WIDTH if icon != "NONE" else 0)
    width = max(80.0, (region.width if region is not None else 300) - margin * scale)
    col = layout.column(align=True)
    col.alert = alert
    col.active = not muted
    lines = text_lines(text, width, context)
    if len(lines) > 1:
        col.scale_y = 0.8
    for i, line in enumerate(lines):
        col.label(text=line, icon=icon if i == 0 else ("BLANK1" if icon != "NONE" else "NONE"))


def muted(layout, text: str, icon: str = "NONE", center: bool = False) -> None:
    row = layout.row()
    row.active = False
    if center:
        row.alignment = "CENTER"
    row.label(text=text, icon=icon)


def text_field(layout, data, name: str, placeholder: str, **kw) -> None:
    if "placeholder" in _PROP_ARGS:  # Blender 4.2 has no placeholders in text fields
        kw["placeholder"] = placeholder
    layout.prop(data, name, **kw)


def fold(layout, flag: str, title: str, summary: str = ""):
    """A fold that starts closed. Returns its body, or None while it is closed."""
    header, body = layout.panel_prop(bpy.context.window_manager.dronegen_ui, flag)
    header.label(text=title)
    if summary:
        sub = header.row()
        sub.alignment = "RIGHT"
        sub.active = False
        sub.label(text=summary)
    return body


# -- blocks ----------------------------------------------------------------------------------
def server_address() -> str:
    try:
        return urllib.parse.urlsplit(prefs.get().server_url).netloc or prefs.get().server_url
    except Exception:  # noqa: BLE001
        return ""


def _is_local() -> bool:
    from .client import is_local

    try:
        return is_local(urllib.parse.urlsplit(prefs.get().server_url).hostname)
    except Exception:  # noqa: BLE001
        return False


def draw_server(layout, context) -> None:
    """The server card: works / not running / error, with the actions that fit the state."""
    server = jobs.STATE.server
    box = layout.box()
    col = box.column()
    if server["ok"]:
        health = server.get("health") or {}
        head = col.row(align=True)
        head.label(text="Сервер работает", icon=icon(*DOT_OK))
        tools = head.row(align=True)
        tools.alignment = "RIGHT"
        tools.operator("dronegen.unload_models", text="", icon="TRASH", emboss=False)
        tools.operator("dronegen.stop_server", text="", icon="QUIT", emboss=False)
        gpu = health.get("gpu") or {}
        if gpu.get("available"):
            total, free = float(gpu.get("total_gb") or 0), float(gpu.get("free_gb") or 0)
            used = max(0.0, total - free)
            col.progress(factor=min(1.0, used / total) if total else 0.0, type="BAR",
                         text=f"Видеопамять {used:.1f} из {total:.0f} ГБ")
            name = str(gpu.get("name", "")).replace("NVIDIA ", "").replace("GeForce ", "")
            loaded = [b.get("title", b.get("name", "")) for kind in ("txt2img", "img2mesh")
                      for b in (health.get("backends") or {}).get(kind, []) if b.get("loaded")]
            muted(col, name + (f" · в памяти {', '.join(loaded)}" if loaded else ""))
        else:
            wrap(col, f"Без видеокарты: {gpu.get('reason', 'нет GPU')}", context, muted=True, boxed=True)
        queue = health.get("queue") or {}
        waiting = int(queue.get("pending") or 0) + (1 if queue.get("running") else 0)
        if waiting:
            muted(col, f"Задач на сервере: {waiting}", icon="SORTTIME")
    elif server["ok"] is False and server.get("offline"):
        col.label(text="Сервер не запущен", icon=icon(*DOT_ERROR))
        muted(col, server_address(), icon="BLANK1")
        if _is_local():
            row = col.row()
            row.scale_y = 1.3
            row.operator("dronegen.start_server", text="Запустить сервер", icon="PLAY")
        else:
            wrap(col, "Запустите сервер на том компьютере; адрес и токен — в настройках аддона", context,
                 muted=True, boxed=True)
    elif server["ok"] is False:
        col.label(text="Сервер ответил ошибкой", icon=icon(*DOT_ERROR))
        wrap(col, server["message"], context, boxed=True)
        col.operator("dronegen.check_server", text="Проверить снова", icon="FILE_REFRESH")
    else:
        wrap(col, server["message"], context, icon=icon(*DOT_WAIT), boxed=True)


def draw_link(layout, context) -> None:
    """Models sent from the Studio with «В Blender»: the switch and the last one that arrived."""
    p = prefs.get()
    layout.prop(p, "studio_link", text="Принимать модели из студии")
    if not p.studio_link:
        return
    name = next((n for n in reversed(link._state["imported"]) if n in context.view_layer.objects), None)
    if name:
        row = layout.row(align=True)
        sub = row.row()
        sub.active = False
        sub.label(text=name, icon="IMPORT")
        row.operator("dronegen.select_object", text="", icon="RESTRICT_SELECT_OFF", emboss=False).name = name


def draw_errors(layout, context) -> None:
    if not jobs.STATE.errors:
        return
    box = layout.box()
    col = box.column()
    head = col.row(align=True)
    head.alert = True
    head.label(text="Что-то пошло не так", icon="ERROR")
    close = head.row(align=True)
    close.alignment = "RIGHT"
    close.operator("dronegen.dismiss", text="", icon="X", emboss=False)
    for message in jobs.STATE.errors[-3:]:
        col.separator(factor=0.4)
        wrap(col, message, context, boxed=True)


def draw_jobs(layout, context, kind: str) -> None:
    """Every running job of this kind and the newest finished one."""
    items = jobs.jobs_of(kind)
    active = [j for j in items if j.active]
    finished = [j for j in items if j.status in ("done", "cancelled")]
    if not active and not finished:
        return
    col = layout.column()
    for job in active:
        muted(col, job.meta.get("subject") or job.title, icon=KIND_ICONS.get(kind, "NONE"))
        row = col.row(align=True)
        row.progress(factor=max(0.0, min(1.0, job.progress)), type="BAR",
                     text=f"{job.stage} · {int(job.progress * 100)}%" if job.progress > 0 else job.stage)
        row.operator("dronegen.cancel_job", text="", icon="X").key = job.key
    if finished:
        job = finished[-1]
        row = col.row(align=True)
        if job.status == "done":
            secs = job.metrics.get("total_seconds")
            row.label(text=job.output + (f" · {secs:.0f} с" if secs else ""), icon="CHECKMARK")
        else:
            row.active = False
            row.label(text=f"Отменено: {job.meta.get('subject') or job.title}", icon="CANCEL")
        row.operator("dronegen.forget_job", text="", icon="X", emboss=False).key = job.key
        for w in job.warnings:
            wrap(col, w, context, icon="INFO", muted=True)


def backend_note(layout, context, kind: str, name: str) -> None:
    info = jobs.backend_info(kind, name)
    if info is not None and not info["available"]:
        wrap(layout, f"Недоступно: {info['reason']}", context, icon="ERROR", alert=True)


def picture_title(path, meta: dict | None = None) -> str:
    meta = gallery.meta(path) if meta is None else meta
    return meta.get("prompt_user") or meta.get("prompt") or path.stem


def picture_facts(meta: dict) -> str:
    parts = [IMAGE_BACKEND_NAMES.get(meta.get("backend"), meta.get("backend", ""))]
    if meta.get("width"):
        parts.append(f"{meta['width']}×{meta.get('height', '?')}")
    if meta.get("style") in STYLE_NAMES:
        parts.append(STYLE_NAMES[meta["style"]])
    if meta.get("seed") is not None:
        parts.append(f"seed {meta['seed']}")
    return " · ".join(p for p in parts if p)


def preview_scale(context) -> float:
    """The gallery preview fills the panel width (in UI units of 20 px)."""
    region = getattr(context, "region", None)
    scale = _ui_scale(context)
    width = (region.width if region is not None else 300) / scale - TEXT_MARGIN
    return max(5.0, min(14.0, width / 20.0))


# -- panels ----------------------------------------------------------------------------------
class DRONEGEN_MT_presets(bpy.types.Menu):
    bl_label = "Готовые промпты"
    bl_idname = "DRONEGEN_MT_presets"

    def draw(self, context):
        for key, label, _prompt in presets.PRESETS:
            self.layout.operator("dronegen.use_preset", text=label).key = key


class DRONEGEN_PT_main(bpy.types.Panel):
    bl_label = "DroneGen"
    bl_idname = "DRONEGEN_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"

    def draw_header(self, context):
        server = jobs.STATE.server
        dots = DOT_OK if server["ok"] else (DOT_ERROR if server["ok"] is False else DOT_WAIT)
        self.layout.label(icon=icon(*dots))

    def draw_header_preset(self, context):
        row = self.layout.row(align=True)
        row.operator("dronegen.check_server", text="", icon="FILE_REFRESH", emboss=False)
        row.operator("preferences.addon_show", text="", icon="PREFERENCES", emboss=False).module = prefs.ADDON_ID

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        row = col.row(align=True)
        row.scale_y = 1.6
        row.operator("dronegen.open_studio", text="Открыть студию", icon="WINDOW", depress=True)
        row.operator("dronegen.open_workspace", text="", icon="WORKSPACE")
        hotkey = studio.hotkey_text()
        muted(col, (f"{hotkey} · " if hotkey else "") + "чат, картинки, 3D-просмотр", center=True)

        draw_server(layout, context)
        draw_link(layout, context)
        draw_errors(layout, context)


class DRONEGEN_PT_image(bpy.types.Panel):
    bl_label = "Картинка"
    bl_idname = "DRONEGEN_PT_image"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"
    bl_parent_id = "DRONEGEN_PT_main"

    def draw_header(self, context):
        self.layout.label(icon="IMAGE_DATA")

    def draw(self, context):
        s = context.scene.dronegen
        layout = self.layout
        row = layout.row(align=True)
        text_field(row, s, "prompt", "Что нарисовать (по-английски)", text="")
        row.menu("DRONEGEN_MT_presets", text="", icon="PRESET")

        grid = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
        for style, text, style_icon in STYLE_BUTTONS:
            grid.prop_enum(s, "image_style", style, text=text, icon=style_icon)
        wrap(layout, STYLE_HINTS.get(s.image_style, ""), context, muted=True)

        w, h = s.image_size()
        body = fold(layout, "image_options", "Параметры", f"{IMAGE_BACKEND_NAMES.get(s.image_backend, '')} · {w}×{h}")
        if body is not None:
            body.use_property_split = True
            body.use_property_decorate = False
            body.prop(s, "image_backend")
            body.prop(s, "size")
            body.prop(s, "steps")
            body.prop(s, "seed")
            if s.image_backend == "sdxl":
                body.prop(s, "negative")
        backend_note(layout, context, "txt2img", s.image_backend)

        row = layout.row()
        row.scale_y = 1.5
        row.operator("dronegen.generate_image", text="Сгенерировать", icon="IMAGE_DATA")
        draw_jobs(layout, context, "txt2img")


class DRONEGEN_PT_gallery(bpy.types.Panel):
    bl_label = "Галерея"
    bl_idname = "DRONEGEN_PT_gallery"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"
    bl_parent_id = "DRONEGEN_PT_main"

    def draw_header(self, context):
        self.layout.label(icon="IMGDISPLAY")

    def draw_header_preset(self, context):
        row = self.layout.row(align=True)
        row.operator("dronegen.refresh_gallery", text="", icon="FILE_REFRESH", emboss=False)
        row.operator("dronegen.open_folder", text="", icon="FILE_FOLDER", emboss=False)

    def draw(self, context):
        layout = self.layout
        path = gallery.selected(context)
        if path is None:
            box = layout.box()
            muted(box, "Пока пусто", icon="IMAGE_DATA")
            wrap(box, "Сгенерированные картинки появятся здесь", context, muted=True, boxed=True)
            return
        meta = gallery.meta(path)
        index, count = gallery.position(path.name)

        row = layout.row()
        row.alignment = "CENTER"
        row.template_icon_view(context.window_manager, "dronegen_gallery", show_labels=True,
                               scale=preview_scale(context), scale_popup=5.0)
        nav = layout.row(align=True)
        nav.operator("dronegen.gallery_step", text="", icon="TRIA_LEFT").delta = -1
        nav.operator("dronegen.gallery_step", text="", icon="TRIA_RIGHT").delta = 1
        mid = nav.row(align=True)
        mid.alignment = "CENTER"
        mid.active = False
        mid.label(text=f"{index} из {count}")
        nav.operator("dronegen.open_image_editor", text="", icon="IMAGE")
        nav.operator("dronegen.delete_image", text="", icon="TRASH").name = path.name

        col = layout.column(align=True)
        wrap(col, picture_title(path, meta), context)
        wrap(col, picture_facts(meta), context, muted=True)


class DRONEGEN_PT_mesh(bpy.types.Panel):
    bl_label = "3D-модель"
    bl_idname = "DRONEGEN_PT_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"
    bl_parent_id = "DRONEGEN_PT_main"

    def draw_header(self, context):
        self.layout.label(icon="MESH_MONKEY")

    def draw(self, context):
        s = context.scene.dronegen
        layout = self.layout
        layout.row(align=True).prop(s, "source", expand=True)
        path = None
        if s.source == "FILE":
            text_field(layout, s, "image_file", "Картинка с диска", text="")
        else:
            path = gallery.selected(context)
            box = layout.box()
            if path is None:
                box.label(text="Выберите картинку в галерее", icon="INFO")
            else:
                meta = gallery.meta(path)
                row = box.row()
                row.template_icon(icon_value=gallery.icon_id(path.name), scale=2.2)
                col = row.column(align=True)
                col.label(text=picture_title(path, meta))
                style = STYLE_NAMES.get(meta.get("style")) or ("без стиля" if meta.get("style") == "none" else "")
                muted(col, " · ".join(x for x in (style, f"{meta.get('width', '?')}×{meta.get('height', '?')}") if x))

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "mesh_backend", text="Способ")
        backend_note(col, context, "img2mesh", s.mesh_backend)
        if s.mesh_backend.startswith("hunyuan"):
            col.prop(s, "mesh_style")
            col.prop(s, "symmetry")
        if s.mesh_backend == "silhouette":
            col.prop(s, "depth")
        col.prop(s, "target_size")
        pic_style = gallery.meta(path).get("style") if path is not None else None
        if s.mesh_backend == "hunyuan_mv" and path is not None and pic_style != "sheet":
            wrap(layout, "Hunyuan3D-mv лучше всего строит по листу из 4 видов (по одной картинке — только спереди)", context,
                 icon="INFO", muted=True)
        elif s.mesh_backend != "hunyuan_mv" and pic_style == "sheet":
            wrap(layout, "На картинке лист из 4 видов: будет взята первая фигура (для всех 4 видов выберите Hunyuan3D-mv)",
                 context, icon="INFO", muted=True)

        body = fold(layout, "mesh_options", "Параметры", f"сетка {s.mc_resolution}")
        if body is not None:
            body.use_property_split = True
            body.use_property_decorate = False
            body.prop(s, "remove_bg")
            body.prop(s, "mc_resolution")
            body.prop(s, "target_faces")

        row = layout.row()
        row.scale_y = 1.5
        row.operator("dronegen.make_3d", text="Сделать 3D-модель", icon="MESH_MONKEY")
        draw_jobs(layout, context, "img2mesh")


class DRONEGEN_PT_anim(bpy.types.Panel):
    bl_label = "Анимация"
    bl_idname = "DRONEGEN_PT_anim"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"
    bl_parent_id = "DRONEGEN_PT_main"

    def draw_header(self, context):
        self.layout.label(icon="ARMATURE_DATA")

    def draw(self, context):
        s = context.scene.dronegen
        layout = self.layout
        mesh = rig.figure_of(context.active_object)
        if mesh is None:
            box = layout.box()
            muted(box, "Выберите модель", icon="RESTRICT_SELECT_OFF")
            wrap(box, "Меш DroneGen, его точки дронов или его скелет", context, muted=True, boxed=True)
            draw_jobs(layout, context, "rig")
            return
        armature = rig.rig_of(mesh)
        layout.label(text=mesh.name, icon="MESH_DATA")

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "rig_plan")
        col.prop(s, "rig_method")
        backend_note(col, context, "rig", s.rig_method)
        if jobs.STATE.server.get("ok") and jobs.backend_info("rig", s.rig_method) is None:
            wrap(col, "Этот метод не установлен на сервере", context, icon="INFO", muted=True)
        col.prop(s, "rig_bind")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("dronegen.build_rig", text="Перестроить скелет" if armature else "Построить скелет",
                     icon="ARMATURE_DATA")
        draw_jobs(layout, context, "rig")
        if armature is None:
            return

        box = layout.box()
        col = box.column()
        data = rig.info(armature)
        col.label(text=armature.name, icon="OUTLINER_OB_ARMATURE")
        wrap(col, rig.describe({**data, "bones": [{"role": b.get("dg_role")} for b in armature.data.bones]}), context,
             muted=True, boxed=True)
        for note in data.get("notes") or []:
            wrap(col, note, context, icon="INFO", muted=True, boxed=True)
        points = rig.drone_points(mesh)
        if not rig.bound(mesh, armature):
            wrap(col, "Модель не привязана к скелету", context, icon="ERROR", alert=True, boxed=True)
        elif points:
            loose = [p.name for p in points if not rig.bound(p, armature)]
            if loose:
                wrap(col, f"Точки дронов не привязаны: {', '.join(loose)} — нажмите «Привязать»", context,
                     icon="ERROR", alert=True, boxed=True)
            else:
                muted(col, "Точки дронов двигаются вместе с костями", icon="CHECKMARK")
        else:
            wrap(col, "Точек дронов ещё нет: после Pointille «Распределить вершины» нажмите «Привязать»", context,
                 icon="INFO", muted=True, boxed=True)

        editing = context.mode == "EDIT_ARMATURE" and context.active_object is armature
        posing = context.mode == "POSE" and context.active_object is armature
        row = col.row(align=True)
        if editing or posing:
            row.operator("dronegen.rig_mode", text="Готово", icon="CHECKMARK", depress=True).mode = "OBJECT"
        else:
            row.operator("dronegen.rig_mode", text="Править", icon="EDITMODE_HLT").mode = "EDIT"
            row.operator("dronegen.rig_mode", text="Поза", icon="POSE_HLT").mode = "POSE"
        row = col.row(align=True)
        row.operator("dronegen.bind_rig", text="Привязать", icon="LINKED")
        row.operator("dronegen.reset_pose", text="", icon="LOOP_BACK")
        row.operator("dronegen.remove_rig", text="", icon="TRASH")


KIND_SUFFIX = {"pose": " · поза", "once": " · разово"}
LAYER_ICONS = {"MOTION": "ARMATURE_DATA", "PATH": "CURVE_PATH", "SHAPE": "MOD_WAVE", "EFFECT": "PARTICLES"}


def _armature(context):
    mesh = rig.figure_of(context.active_object)
    return rig.rig_of(mesh) if mesh is not None else None


class DRONEGEN_MT_motions(bpy.types.Menu):
    bl_label = "Движения"
    bl_idname = "DRONEGEN_MT_motions"

    def draw(self, context):
        layout = self.layout
        armature = _armature(context)
        if not motion.CATALOG:
            layout.operator("dronegen.fetch_motions", text="Загрузить движения с сервера", icon="FILE_REFRESH")
            return
        groups: dict[str, list] = {}
        for key, m in (motion.fitting(armature) if armature is not None else motion.CATALOG.items()):
            groups.setdefault(m.get("group") or "Прочее", []).append((key, m))
        if not groups:
            layout.label(text="Для этого скелета движений нет", icon="INFO")
            return
        row = layout.row()
        for group, items in groups.items():
            col = row.column()
            col.label(text=group)
            for key, m in items:
                col.operator("dronegen.set_motion", text=m["title"] + KIND_SUFFIX.get(m["kind"], "")).key = key


class DRONEGEN_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon_value, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        sub = row.row(align=True)
        sub.active = item.enabled
        sub.prop(item, "name", text="", emboss=False, icon=LAYER_ICONS.get(item.kind, "ARMATURE_DATA"))
        frames = row.row()
        frames.alignment = "RIGHT"
        frames.active = False
        frames.label(text=f"{item.start}–{item.end}")


def draw_layer(layout, context, layer) -> None:
    """Settings of one animation layer: a motion (a menu of what the skeleton can do), a flight, an effect of shape
    or an effect with drones of its own."""
    box = layout.box()
    col = box.column()
    if layer.kind == "MOTION":
        m = motion.CATALOG.get(layer.motion) or {}
        col.menu("DRONEGEN_MT_motions", text=motion.title(layer), icon="ARMATURE_DATA")
        if m.get("about"):
            wrap(col, m["about"], context, muted=True, boxed=True)
    elif layer.kind == "PATH":
        col.row(align=True).prop(layer, "path_shape", expand=True)
    elif layer.kind == "SHAPE":
        col.prop(layer, "shape_fx", text="")
    else:
        col.prop(layer, "effect", text="")
        about = layer.bl_rna.properties["effect"].enum_items[layer.effect].description
        wrap(col, about + ". Дроны ждут в резерве за фигурой и прилетают к началу", context, muted=True, boxed=True)
    kind = (motion.CATALOG.get(layer.motion) or {}).get("kind") if layer.kind == "MOTION" else None
    row = box.row(align=True)  # pairs keep a label each: a split row would show only the first
    row.prop(layer, "start", text="Начало")
    if kind != "once":
        row.prop(layer, "end", text="Конец")
    col = box.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    if layer.kind == "MOTION":
        if kind != "pose":
            col.prop(layer, "period")
        col.prop(layer, "amplitude")
        col.prop(layer, "strength")
        if "side" in ((motion.CATALOG.get(layer.motion) or {}).get("params") or {}):
            col.prop(layer, "side")
    elif layer.kind == "PATH":
        if layer.path_shape == "CURVE":
            col.prop(layer, "curve")
        else:
            col.prop(layer, "radius")
        if layer.path_shape != "LINE":
            col.prop(layer, "laps")
    elif layer.kind == "SHAPE":
        if layer.shape_fx != "SCATTER":
            col.prop(layer, "period")
        col.prop(layer, "fx_size")
        col.prop(layer, "strength")
    else:
        if layer.effect == "model":
            col.prop(layer, "model")
        if layer.effect != "trail":
            col.prop(layer, "size")
            col.prop(layer, "anchor")
            if layer.anchor == "BONE":
                col.prop(layer, "bone")
            col.prop(layer, "offset")
        else:
            col.prop(layer, "bone")
        col.prop(layer, "count")
        col.prop(layer, "travel")
    row = box.row(align=True)
    if layer.kind == "EFFECT":
        row.prop(layer, "fade_out", text="Держится после" if layer.effect == "trail" else "Выход")
    else:
        row.prop(layer, "fade_in", text="Вход")
        row.prop(layer, "fade_out", text="Выход")


class DRONEGEN_PT_layers(bpy.types.Panel):
    bl_label = "Слои анимации"
    bl_idname = "DRONEGEN_PT_layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"
    bl_parent_id = "DRONEGEN_PT_anim"

    @classmethod
    def poll(cls, context):
        return _armature(context) is not None

    def draw_header(self, context):
        self.layout.label(icon="NLA")

    def draw(self, context):
        layout = self.layout
        armature = _armature(context)
        row = layout.row()
        row.template_list("DRONEGEN_UL_layers", "", armature, "dronegen_layers", armature, "dronegen_layer", rows=3)
        side = row.column(align=True)
        side.operator("dronegen.layer_add", text="", icon="ADD").kind = "MOTION"
        side.operator("dronegen.layer_add", text="", icon="CURVE_PATH").kind = "PATH"
        side.operator("dronegen.layer_add", text="", icon="MOD_WAVE").kind = "SHAPE"
        side.operator("dronegen.layer_add", text="", icon="PARTICLES").kind = "EFFECT"
        side.operator("dronegen.layer_remove", text="", icon="REMOVE")
        side.separator()
        side.operator("dronegen.layer_move", text="", icon="TRIA_UP").step = -1
        side.operator("dronegen.layer_move", text="", icon="TRIA_DOWN").step = 1
        layers = armature.dronegen_layers
        if not layers:
            wrap(layout, "Добавьте движение (+) или полёт: слои складываются — крылья машут, ноги скачут, фигура "
                         "летит по кругу, всё одновременно", context, icon="INFO", muted=True)
        else:
            draw_layer(layout, context, layers[min(armature.dronegen_layer, len(layers) - 1)])
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("dronegen.apply_layers", text="Применить", icon="PLAY")
        row.operator("dronegen.clear_layers", text="", icon="X")
        draw_jobs(layout, context, "anim")
        draw_jobs(layout, context, "trail")
        count = fx.budget(armature, context.scene)
        total = count["figure"] + count["effects"]
        text = f"Дронов: фигура {count['figure']}" + (f" + эффекты {count['effects']} = {total}" if count["effects"] else "")
        if count["fleet"]:
            text += f" · в Skybrush {count['fleet']}"
        wrap(layout, text, context, icon="ERROR" if count["fleet"] and total > count["fleet"] else "NONE",
             alert=bool(count["fleet"] and total > count["fleet"]), muted=not (count["fleet"] and total > count["fleet"]))

class DRONEGEN_PT_drones_check(bpy.types.Panel):
    bl_label = "Проверка дронов"
    bl_idname = "DRONEGEN_PT_drones_check"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DroneGen"
    bl_parent_id = "DRONEGEN_PT_anim"

    @classmethod
    def poll(cls, context):
        return rig.figure_of(context.active_object) is not None

    def draw_header(self, context):
        self.layout.label(icon="CHECKMARK")

    def draw(self, context):
        layout = self.layout
        armature = _armature(context)
        lim = safety.limits(context.scene)
        wrap(layout, f"Пределы: {lim['xy']:g} м/с по горизонтали, подъём {lim['up']:g}, спуск {lim['down']:g} м/с, "
                     f"ускорение {lim['acceleration']:g} м/с², расстояние {lim['distance']:g} м — {lim['source']}",
             context, muted=True)
        row = layout.row()
        row.scale_y = 1.3
        row.operator("dronegen.check_drones", text="Проверить дронов", icon="VIEWZOOM")
        report = safety.last(armature)
        if report is None:
            return
        box = layout.box()
        col = box.column()
        worst = report["worst"]
        if report["ok"]:
            col.label(text="Дроны справятся", icon="CHECKMARK")
        for text in report["problems"]:
            wrap(col, text, context, icon="ERROR", alert=True, boxed=True)
        facts = [f"{report['drones']} точек, кадры {report['frames'][0]}–{report['frames'][1]}"]
        if "xy" in worst:
            facts.append(f"скорость до {worst['xy']['value']:.1f} м/с, подъём {worst['up']['value']:.1f}, "
                         f"спуск {worst['down']['value']:.1f}")
        facts.append(f"ближе всего {worst['distance']['value']:.2f} м")
        wrap(col, " · ".join(facts), context, muted=True, boxed=True)
        if report["slow"] > 1.05 and armature is not None:
            k = math.ceil(report["slow"] * 10) / 10
            col.operator("dronegen.slow_down", text=f"Замедлить в {k:g} раза", icon="SORTTIME").factor = k


classes = (DRONEGEN_MT_presets, DRONEGEN_PT_main, DRONEGEN_PT_image, DRONEGEN_PT_gallery, DRONEGEN_PT_mesh,
           DRONEGEN_PT_anim, DRONEGEN_MT_motions, DRONEGEN_UL_layers, DRONEGEN_PT_layers, DRONEGEN_PT_drones_check)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
