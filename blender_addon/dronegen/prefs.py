"""Add-on preferences: where the server is, where results are cached, how to start it."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

ADDON_ID = __package__


def _install_info() -> dict:
    """Written next to the add-on by install.py: where the server is installed."""
    try:
        return json.loads(Path(__file__).with_name("install_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


INFO = _install_info()
DEFAULT_URL = INFO.get("server_url", "http://127.0.0.1:8188")


class DroneGenPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    server_url: StringProperty(
        name="Адрес сервера",
        description="Где работает сервер DroneGen. Для другого компьютера: http://<ip>:8188",
        default=DEFAULT_URL,
    )
    token: StringProperty(
        name="Токен",
        description="Нужен, только если сервер запущен с --token (например, на другом компьютере)",
        subtype="PASSWORD",
        default="",
    )
    cache_dir: StringProperty(
        name="Папка кэша",
        description="Куда сохранять картинки и меши. Пусто — папка аддона в профиле Blender",
        subtype="DIR_PATH",
        default="",
    )
    server_home: StringProperty(
        name="Папка сервера",
        description="Папка установки сервера (runtime с venv): нужна кнопке «Запустить сервер»",
        subtype="DIR_PATH",
        default=INFO.get("home", ""),
    )
    timeout: IntProperty(name="Таймаут запросов, с", default=15, min=2, max=300)
    studio_link: BoolProperty(
        name="Принимать модели из студии",
        description="Модели, отправленные в студии кнопкой «В Blender», сами появляются в коллекции DroneGen",
        default=True,
    )

    def draw(self, context):
        from . import studio, ui

        layout = self.layout
        ui.draw_server(layout, context)

        box = layout.box()
        box.label(text="Подключение", icon="URL")
        col = box.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(self, "server_url")
        col.prop(self, "token")
        col.prop(self, "timeout")
        col.operator("dronegen.check_server", icon="FILE_REFRESH")

        box = layout.box()
        box.label(text="Сервер на этом компьютере", icon="DESKTOP")
        col = box.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(self, "server_home")
        row = col.row(align=True)
        row.operator("dronegen.start_server", icon="PLAY")
        row.operator("dronegen.stop_server", icon="QUIT")
        row.operator("dronegen.unload_models", icon="TRASH")

        box = layout.box()
        box.label(text="Студия", icon="WINDOW")
        col = box.column()
        col.prop(self, "studio_link")
        studio.draw_hotkey(col)

        box = layout.box()
        box.label(text="Файлы", icon="FILE_FOLDER")
        col = box.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(self, "cache_dir")
        col.operator("dronegen.open_folder", icon="FILE_FOLDER")


def get():
    return bpy.context.preferences.addons[ADDON_ID].preferences


def cache_dir() -> Path:
    raw = get().cache_dir
    if raw:
        path = Path(bpy.path.abspath(raw))
    else:
        try:
            path = Path(bpy.utils.extension_path_user(ADDON_ID, path="cache", create=True))
        except ValueError:  # installed as a legacy add-on, not as an extension
            path = Path(bpy.utils.user_resource("SCRIPTS", path="dronegen_cache", create=True))
    path.mkdir(parents=True, exist_ok=True)
    return path


def client():
    from .client import Client

    p = get()
    return Client(p.server_url, p.token, p.timeout)


def register():
    bpy.utils.register_class(DroneGenPreferences)
    try:
        from .server_proc import venv_python

        p = get()
        installed = _install_info().get("home", "")
        if installed and (not p.server_home or not venv_python(Path(p.server_home).expanduser().resolve()).is_file()):
            p.server_home = installed
    except Exception:  # noqa: BLE001
        pass


def unregister():
    bpy.utils.unregister_class(DroneGenPreferences)
