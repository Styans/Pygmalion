"""Baiterek: the tower in Astana, a white "tree of life" holding a golden egg.

A flared base and a slender white trunk that opens into curved lattice branches cradling a golden
sphere (its diameter is ~23 % of the height). All sizes are fractions of the height.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import shapes as S
from .registry import Param, Template, register

WHITE = (0.94, 0.95, 0.97)
GOLD = (0.96, 0.76, 0.26)
GREY = (0.62, 0.64, 0.68)

# the branches in the plane through the axis: (distance from the axis, height), fractions of H
BRANCH = [(0.020, 0.58), (0.055, 0.65), (0.105, 0.72), (0.150, 0.79), (0.158, 0.85), (0.115, 0.92), (0.045, 0.975)]


def build(v: dict):
    H = v["height"]
    n = int(v["branches"])
    egg_r = v["sphere_ratio"] * H / 2
    parts = [S.solid(S.lathe([(0.075 * H * v["base"], 0.0), (0.058 * H * v["base"], 0.025 * H), (0.026 * H, 0.09 * H),
                              (0.013 * H, 0.18 * H), (0.011 * H, 0.45 * H), (0.016 * H, 0.58 * H)], 48), WHITE)]
    parts.append(S.solid(S.revolve_closed([(0.08 * H * v["base"], 0.0), (0.095 * H * v["base"], 0.0),
                                           (0.095 * H * v["base"], 0.012 * H), (0.08 * H * v["base"], 0.012 * H)], 64), GREY))
    r = np.array([b[0] for b in BRANCH]) * H
    y = np.array([b[1] for b in BRANCH]) * H
    s = np.linspace(0, 1, len(BRANCH))
    sr, sy = PchipInterpolator(s, r), PchipInterpolator(s, y)
    ss = np.linspace(0, 1, 40)
    rho, yy = sr(ss), sy(ss)
    for k in range(n):
        phi = 2 * np.pi * k / n
        path = np.stack([rho * np.cos(phi), yy, rho * np.sin(phi)], axis=1)
        parts.append(S.solid(S.tube(path, 0.0034 * H, 6), WHITE))
    for frac in (0.20, 0.45, 0.70, 0.88):  # rings joining the branches
        i = int(frac * (len(ss) - 1))
        parts.append(S.solid(S.revolve_closed([(rho[i] - 0.0022 * H, yy[i] - 0.0022 * H), (rho[i] + 0.0022 * H, yy[i] - 0.0022 * H),
                                               (rho[i] + 0.0022 * H, yy[i] + 0.0022 * H), (rho[i] - 0.0022 * H, yy[i] + 0.0022 * H)], 64), WHITE))
    parts.append(S.solid(S.sphere((0, 0.795 * H, 0), egg_r, 40), GOLD))
    parts.append(S.solid(S.lathe([(0.009 * H, 0.975 * H), (0.004 * H, 0.99 * H), (0.001 * H, H)], 12), WHITE))
    return S.join(parts)


register(Template(
    "baiterek", "Байтерек",
    {
        "height": Param(1.0, 0.3, 3.0, "высота", "м"),
        "sphere_ratio": Param(0.227, 0.15, 0.32, "диаметр золотого шара к высоте (у настоящего 22 м из 97)"),
        "branches": Param(16, 6, 28, "число ветвей-решёток", integer=True),
        "base": Param(1.0, 0.5, 2.0, "ширина основания (1 — как у настоящего)"),
    },
    build,
    "Байтерек: расширяющееся основание, тонкий белый ствол, изогнутые ветви-решётки и золотой шар.",
    ("байтерек", "baiterek"),
))
