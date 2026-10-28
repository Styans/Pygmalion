"""Dombra: the Kazakh two-string lute.

Teardrop body: a flat soundboard and a deep back of 7-11 flat staves (facets), the outline from a
smooth curve; a neck with frets spaced like a lute's; a flat head tilted back with exactly two
pegs, one on each side; a bridge, a button for the strings, two strings, a small soundhole.
Real sizes: about 1 m long, the body ~36 cm x 24 cm x 10 cm.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import shapes as S
from .registry import Param, Template, register

WOOD_BACK = (0.60, 0.36, 0.19)
WOOD_FRONT = (0.88, 0.72, 0.49)
NECK = (0.45, 0.28, 0.15)
FINGERBOARD = (0.20, 0.12, 0.08)
HEAD = (0.38, 0.22, 0.12)
PEG = (0.72, 0.52, 0.30)
BONE = (0.93, 0.89, 0.78)
STRING = (0.92, 0.92, 0.86)
BLACK = (0.03, 0.03, 0.03)

OUTLINE_T = [0.0, 0.03, 0.10, 0.22, 0.38, 0.55, 0.72, 0.88, 1.0]
OUTLINE_W = [0.0, 0.50, 0.78, 0.95, 1.00, 0.92, 0.70, 0.38, 0.17]  # of the maximum half-width; the last is replaced
HEAD_LEN = 0.12
FLAT_POINTS = 6  # points across the flat soundboard of each section


def d_section(y: float, half_w: float, depth: float, facets: int, z0: float = 0.0) -> np.ndarray:
    """A D-shaped section: the flat front at z=z0, the back an arc (or ``facets`` flat staves) behind it."""
    k = max(facets, 0) or 28
    theta = np.arange(k + 1) * np.pi / k
    back = np.stack([half_w * np.cos(theta), np.full_like(theta, y), z0 - depth * np.sin(theta)], axis=1)
    xs = np.linspace(-half_w, half_w, FLAT_POINTS + 2)[1:-1]
    front = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z0)], axis=1)
    return np.vstack([back, front])


def build(v: dict):
    L, Lb, Wb = v["length"], v["body_length"], v["body_width"]
    D = v["body_depth"] * Wb
    nw = v["neck_width"]
    facets = int(v["back_facets"])
    y_nut = L - HEAD_LEN
    y_bridge = 0.075 * Lb / 0.36 + 0.01

    # -- body: sections of the teardrop outline ------------------------------------------------
    widths = np.array(OUTLINE_W)
    widths[-1] = min(0.30, max(0.1, nw / Wb))
    outline = PchipInterpolator(OUTLINE_T, widths)
    t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 56))  # denser near both ends
    t = np.clip(t, 0.004, 1.0)
    half = np.maximum(Wb / 2 * outline(t), 0.0015)
    depth = np.maximum(D * (half / (Wb / 2)) ** 0.85, 0.001)
    body = S.loft([d_section(tt * Lb, w, d, facets) for tt, w, d in zip(t, half, depth)])
    body = S.recolor(body, lambda p: np.where((p[:, 2] > -1e-4)[:, None], WOOD_FRONT, WOOD_BACK))

    parts = [body]

    # -- neck and fingerboard ----------------------------------------------------------------
    y0 = Lb - 0.03
    ys = np.linspace(y0, y_nut, 12)
    neck_half = np.linspace(nw / 2, nw / 2 * 0.86, len(ys))
    neck = S.loft([d_section(y, w, 0.017, 0, z0=0.0) for y, w in zip(ys, neck_half)])
    parts.append(S.solid(neck, NECK))
    board_w = nw * 0.96
    parts.append(S.solid(S.box((0, (Lb - 0.02 + y_nut) / 2, 0.0025), (board_w, y_nut - Lb + 0.02, 0.005)), FINGERBOARD))

    # -- frets: spaced like a lute's (equal temperament) between the nut and the bridge --------
    scale = y_nut - y_bridge
    for k in range(1, int(v["frets"]) + 1):
        y = y_nut - scale * (1 - 2 ** (-k / 12))
        if y < Lb - 0.015:  # frets stop at the body
            break
        parts.append(S.solid(S.box((0, y, 0.0058), (nw * 0.98 + 0.004, 0.0028, 0.0034)), BONE))

    # -- bridge, string button, soundhole ------------------------------------------------------
    parts.append(S.solid(S.box((0, y_bridge, 0.009), (0.07, 0.007, 0.018)), BONE))
    y_button = 0.028
    parts.append(S.solid(S.box((0, y_button, 0.007), (0.018, 0.022, 0.014)), HEAD))
    if v["soundhole"] >= 0.5:
        parts.append(S.solid(S.cylinder((0, 0.60 * Lb, -0.0008), (0, 0.60 * Lb, 0.0012), 0.016 * Wb / 0.24, 28), BLACK))

    # -- head with exactly two pegs, tilted back -------------------------------------------------
    hw, ht = 0.05, 0.020
    head_parts = [
        S.solid(S.box((0, y_nut + (HEAD_LEN - hw / 2) / 2, -0.0035), (hw, HEAD_LEN - hw / 2, ht)), HEAD),
        S.solid(S.cylinder((0, y_nut + HEAD_LEN - hw / 2, -0.0035 - ht / 2), (0, y_nut + HEAD_LEN - hw / 2, -0.0035 + ht / 2),
                           hw / 2, 28), HEAD),
    ]
    for side, along in ((-1, 0.072), (1, 0.040)):  # one peg on each side, at different heights
        y = y_nut + along
        x0 = side * hw / 2
        head_parts.append(S.solid(S.cylinder((x0, y, -0.0035), (side * (hw / 2 + 0.038), y, -0.0035), 0.0065, 16), PEG))
        head_parts.append(S.solid(S.cylinder((side * (hw / 2 + 0.034), y, -0.0035), (side * (hw / 2 + 0.056), y, -0.0035), 0.0105, 20), PEG))
    parts += [S.rotate(p, -v["head_tilt"], (1, 0, 0), (0, y_nut, 0)) for p in head_parts]
    nut_z = 0.007

    # -- two strings ------------------------------------------------------------------------------
    for x in (-0.0065, 0.0065):
        path = [(x, y_button + 0.004, 0.015), (x, y_bridge, 0.0195), (x, y_nut, nut_z)]
        parts.append(S.solid(S.tube(np.array(path), 0.0006, 6), STRING))

    return S.join(parts)


register(Template(
    "dombra", "Домбра",
    {
        "length": Param(1.00, 0.70, 1.30, "общая длина", "м"),
        "body_length": Param(0.36, 0.24, 0.50, "длина корпуса", "м"),
        "body_width": Param(0.24, 0.16, 0.34, "ширина корпуса", "м"),
        "body_depth": Param(0.42, 0.25, 0.60, "толщина корпуса к его ширине"),
        "back_facets": Param(9, 0, 13, "число граней-клёпок на спинке (0 — гладкая)", integer=True),
        "neck_width": Param(0.040, 0.025, 0.060, "ширина грифа", "м"),
        "frets": Param(14, 0, 24, "число ладов", integer=True),
        "head_tilt": Param(12, 0, 30, "наклон головки назад", "°"),
        "soundhole": Param(1, 0, 1, "звуковое отверстие (1 — есть, 0 — нет)", integer=True),
    },
    build,
    "Казахская двухструнная домбра: каплевидный корпус с гранёной спинкой, длинный гриф с ладами, головка "
    "ровно с двумя колками, две струны.",
    ("домбра", "домбыра", "dombra", "dombyra"),
))
