"""Skeletons for animation: a mesh -> named bones (``fit``), whatever builds the tree through the body.

    vertices, faces (Blender object space, Z up) -> voxels -> curve skeleton (``skeleton``)
        -> parts and roles (``roles``) -> bones (``chains``)

The result is plain JSON for the Blender add-on: bones with head, tail, parent, role and side; the body's
axes; what was found and what is missing. A preview sheet (three views, bones coloured by role) shows it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import chains, roles, skeleton
from .roles import BODY_PLANS, hints_of  # noqa: F401

PLAN_TITLES = {  # what the Blender panel and the chat call the body plans
    "auto": "Авто — по форме модели",
    "quadruped": "Четвероногое (конь, барс)",
    "winged_quadruped": "Крылатое четвероногое (тулпар)",
    "bird": "Птица (беркут)",
    "biped": "Двуногое с руками (человек, персонаж)",
    "winged_biped": "Двуногое с руками и крыльями (ангел)",
    "snake": "Без конечностей (змея, рыба)",
    "rigid": "Предмет без частей (домбра, машина)",
}

COLORS = {
    "root": (120, 120, 120), "spine": (30, 30, 30), "neck": (90, 60, 160), "head": (150, 40, 200),
    "tail": (150, 100, 40), "leg_front": (210, 40, 40), "leg_hind": (40, 120, 220), "leg": (210, 40, 40),
    "arm": (230, 140, 20), "wing": (20, 160, 90), "prop": (220, 30, 140), "extra": (130, 130, 130),
}


def fit(vertices, faces, plan=None, resolution: int = skeleton.RESOLUTION) -> tuple[dict, Image.Image]:
    """Bones for a mesh; ``plan``: a body plan name (``BODY_PLANS``), hints {"legs": 4, ...} or None (automatic)."""
    hints = hints_of(plan)
    vol = skeleton.volume(vertices, faces, resolution)
    tree = skeleton.tree(vol)
    body = roles.label(tree, vol, hints)
    bone_list = chains.bones(tree, body)
    result = {
        "bones": chains.as_dicts(bone_list),
        "frame": {"forward": body.forward.round(5).tolist(), "left": body.left.round(5).tolist(),
                  "up": body.up.round(5).tolist()},
        "upright": body.upright,
        "rigid": body.rigid,
        "ground": round(body.ground, 5),
        "height": round(body.height, 5),
        "size": round(body.size, 5),
        "parts": chains.summary(body),
        "notes": body.notes,
        "plan": plan if isinstance(plan, str) else ("hints" if plan else "auto"),
    }
    return result, preview(vol, bone_list)


def preview(vol: skeleton.Volume, bone_list, cell: int = 3, height: int = 300) -> Image.Image:
    """Front, side and top views: the silhouette in grey, the bones as coloured lines with joints."""
    grid = vol.grid
    views = ((0, 2), (1, 2), (0, 1))  # (horizontal axis, vertical axis): front (X, Z), side (Y, Z), top (X, Y)
    images = []
    for h, v in views:
        sil = grid.any(axis=3 - h - v)
        if h > v:
            sil = sil.T
        img = Image.fromarray(np.where(sil.T[::-1], 215, 255).astype(np.uint8)).convert("RGB")
        img = img.resize((img.width * cell, img.height * cell), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        rows = sil.shape[1]

        def px(p):
            ijk = (np.asarray(p) - vol.origin) / vol.pitch
            return ijk[h] * cell + cell / 2, (rows - 1 - ijk[v]) * cell + cell / 2

        for bone in bone_list:
            a, b = px(bone.head), px(bone.tail)
            color = COLORS.get(bone.role, (0, 0, 0))
            draw.line([a, b], fill=color, width=3 if bone.deform else 1)
            draw.ellipse([a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3], fill=color)
        images.append(img)
    w = sum(i.width for i in images) + 10 * (len(images) - 1)
    sheet = Image.new("RGB", (w, max(i.height for i in images)), "white")
    x = 0
    for img in images:
        sheet.paste(img, (x, 0))
        x += img.width + 10
    if sheet.height > height * 2:
        k = height * 2 / sheet.height
        sheet = sheet.resize((int(sheet.width * k), int(sheet.height * k)), Image.LANCZOS)
    return sheet
