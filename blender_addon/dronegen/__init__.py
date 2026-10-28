"""DroneGen: text -> image -> 3D mesh for drone show figures, through a local server.

Alt+Shift+D opens the Studio (chat, projects, 3D preview); models sent with «В Blender» arrive here.

The add-on itself only speaks HTTP (urllib, standard library); every model runs in the
server process. Requests run in background threads and their results are applied on the
main thread by a timer, so Blender never freezes while a model works.

Imported meshes land in the "DroneGen" collection, active and selected, ready for a
formation step (e.g. Pointille's "Distribute vertices" or a future operator here).
"""

bl_info = {
    "name": "DroneGen",
    "author": "Tyan Stanislav",
    "version": (0, 8, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > DroneGen",
    "description": "Промпт → картинка → 3D-меш для фигур дрон-шоу (локальный сервер)",
    "category": "Import-Export",
}

from . import gallery, jobs, link, operators, prefs, props, studio, ui  # noqa: E402

_modules = (prefs, props, gallery, operators, ui, jobs, link, studio)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
