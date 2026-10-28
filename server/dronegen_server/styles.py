"""Picture styles: how the subject is drawn so that the picture suits the 3D method.

* ``flat``   - a bold flat silhouette: for «Силуэт → объём» (and drone shows seen from afar);
* ``render``  - a shaded 3D render: shading tells the 3D models the depth (TripoSR, Hunyuan3D);
* ``sheet``  - front | left | back | right on one canvas: split into cards for Hunyuan3D-2mv, the
  most precise way (docs/BENCHMARKS.md, «Точность 3D»).

The Blender add-on keeps a copy of these strings (blender_addon/dronegen/presets.py);
tests/test_styles.py checks that both stay the same.
"""

from __future__ import annotations

FLAT = "single object, centered, plain white background, bold simple silhouette, flat colors"
RENDER = (
    "clean 3D product render, entire object visible with margins, soft even studio lighting, "
    "soft shadows, matte materials, light gray seamless background, sharp focus"
)
SHEET = (
    "character model sheet, four views of the same object in one row, equal spacing, same scale, same height, "
    "strict orthographic projection, the camera turns exactly 90 degrees between the views: "
    "1) FRONT: the object faces the camera head-on, symmetric; "
    "2) LEFT SIDE: pure profile, the front of the object points to the left edge of the picture; "
    "3) BACK: seen exactly from behind; "
    "4) RIGHT SIDE: pure profile, the front of the object points to the right edge of the picture. "
    "The object: {subject}. Standing upright, " + RENDER
)
NEGATIVE = "photo, background scenery, text, watermark, frame, multiple objects, cropped"

STYLES = {
    "flat": {"title": "Плоский (для силуэта)", "mesh": ("silhouette",), "size": (768, 768)},
    "render": {"title": "Объёмный (для 3D)", "mesh": ("hunyuan", "triposr"), "size": (768, 1024)},
    "sheet": {"title": "Лист из 4 видов (карточки)", "mesh": ("hunyuan_mv",), "size": (1920, 576)},
}


def build(subject: str, style: str = "render", view: str = "front view") -> str:
    """The full English prompt for ``subject`` (a description of the object) in ``style``.

    ``view`` is added unless the subject already names one ("side view", "top view"...).
    """
    subject = subject.strip().rstrip(".")
    if " view" in f" {subject.lower()}":
        view = ""
    view = f", {view}" if view else ""
    if style == "flat":
        return f"{subject}{view}, {FLAT}"
    if style == "render":
        return f"{subject}{view}, {RENDER}"
    if style == "sheet":
        return SHEET.format(subject=subject)
    raise ValueError(f"unknown style {style!r}: {', '.join(STYLES)}")


def for_mesh(backend: str) -> str:
    """The style that suits a 3D backend best."""
    for name, info in STYLES.items():
        if backend in info["mesh"]:
            return name
    return "render"
