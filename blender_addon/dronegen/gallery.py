"""Generated pictures: PNG files in ``<cache>/images`` with a JSON sidecar, thumbnails via previews.

The selection is kept by file name, not by position: new pictures go first and deleted ones
leave, and the selected picture must stay the same through both.
"""

from __future__ import annotations

import json
from pathlib import Path

import bpy
import bpy.utils.previews
from bpy.props import EnumProperty

from . import prefs

MAX_ITEMS = 200
_pcoll = None
# Blender reads enum items lazily: the list returned by the callback must stay referenced,
# otherwise the strings are freed and Blender can crash.
_items: list = []
_current = ""  # the selected file name


def folder() -> Path:
    d = prefs.cache_dir() / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta(path: Path) -> dict:
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def refresh() -> None:
    global _items
    if _pcoll is None:
        return
    files = sorted(folder().glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:MAX_ITEMS]
    items = []
    for i, f in enumerate(files):
        if f.name not in _pcoll:
            _pcoll.load(f.name, str(f), "IMAGE")
        m = meta(f)
        label = (m.get("prompt_user") or m.get("prompt") or f.stem)[:48]
        items.append((f.name, label, m.get("prompt", f.name), _pcoll[f.name].icon_id, i))
    _items = items or [("NONE", "Пока пусто", "Сгенерируйте картинку", 0, 0)]


def _enum_items(self, context):
    return _items


def _index(name: str) -> int:
    return next((i for i, item in enumerate(_items) if item[0] == name), 0)


def _get(self) -> int:
    return _index(_current)  # a deleted picture falls back to the newest one


def _set(self, value: int) -> None:
    global _current
    if 0 <= value < len(_items):
        _current = _items[value][0]


def names() -> list[str]:
    return [item[0] for item in _items if item[0] != "NONE"]


def select(name: str) -> None:
    global _current
    _current = name


def step(delta: int) -> None:
    """Select the picture ``delta`` places away (wraps around)."""
    files = names()
    if files:
        select(files[(_index(_current) + delta) % len(files)])


def position(name: str) -> tuple[int, int]:
    """(number of the picture counting from 1, how many there are)."""
    return _index(name) + 1, len(names())


def selected(context) -> Path | None:
    name = context.window_manager.dronegen_gallery
    if not name or name == "NONE":
        return None
    path = folder() / name
    return path if path.is_file() else None


def icon_id(name: str) -> int:
    return _pcoll[name].icon_id if _pcoll is not None and name in _pcoll else 0


def add(path: Path, info: dict) -> str:
    """Register a downloaded picture and select it."""
    path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    refresh()
    select(path.name)
    return path.name


def remove(name: str) -> None:
    path = folder() / name
    for p in (path, path.with_suffix(".json")):
        if p.is_file():
            p.unlink()
    index = _index(name)
    refresh()
    files = names()
    if files:  # the neighbour takes the place of the deleted picture
        select(files[min(index, len(files) - 1)])


def register():
    global _pcoll
    _pcoll = bpy.utils.previews.new()
    bpy.types.WindowManager.dronegen_gallery = EnumProperty(
        name="Галерея", description="Нажмите, чтобы выбрать картинку", items=_enum_items, get=_get, set=_set,
    )
    try:
        refresh()
    except Exception as e:  # noqa: BLE001 - a broken cache folder must not block the add-on
        print("DroneGen: gallery refresh failed:", e)


def unregister():
    global _pcoll, _items, _current
    del bpy.types.WindowManager.dronegen_gallery
    if _pcoll is not None:
        bpy.utils.previews.remove(_pcoll)
    _pcoll, _items, _current = None, [], ""
