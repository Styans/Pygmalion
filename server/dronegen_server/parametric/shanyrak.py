"""Shanyrak: the round wooden crown of a yurt.

A thick ring, two groups of parallel arched laths crossing at right angles (3 + 3 by default,
the second group lies over the first) that bulge into a low dome, and straight roof poles (uyk)
radiating from the ring like sun rays. Seen from above, as on the emblem of Kazakhstan.
"""

from __future__ import annotations

import numpy as np

from . import shapes as S
from .registry import Param, Template, register

WOOD = (0.50, 0.28, 0.14)
WOOD_LIGHT = (0.84, 0.70, 0.48)
GOLD = (0.92, 0.72, 0.22)
RED = (0.62, 0.16, 0.10)


def _arch(x_end: float, dome: float, y0: float, along: str, offset: float, samples: int = 31) -> np.ndarray:
    s = np.linspace(-1.0, 1.0, samples)
    h = y0 + dome * (1 - s**2)
    if along == "x":
        return np.stack([s * x_end, h, np.full_like(s, offset)], axis=1)
    return np.stack([np.full_like(s, offset), h, s * x_end], axis=1)


def build(v: dict):
    R = v["diameter"] / 2
    rw, h = v["ring_width"], 0.07
    r_in = R - rw
    parts = [
        S.solid(S.revolve_closed([(r_in, -h / 2), (R, -h / 2), (R, h / 2), (r_in, h / 2)], 96), WOOD),
        S.solid(S.revolve_closed([(R + 0.002, -0.012), (R + 0.012, -0.012), (R + 0.012, 0.012), (R + 0.002, 0.012)], 96), GOLD),
        S.solid(S.revolve_closed([(r_in - 0.012, h / 2 - 0.004), (r_in, h / 2 - 0.004), (r_in, h / 2 + 0.008),
                                  (r_in - 0.012, h / 2 + 0.008)], 96), RED),
    ]
    n = int(v["arches"])
    spacing = 0.62 * r_in / max(n - 1, 1) * min(1.0, (n - 1) / 2) if n > 1 else 0.0
    spacing = min(spacing, 0.30 * r_in)
    lath_w, lath_h = 0.05 * R / 0.9 if R > 0.9 else 0.05, 0.026
    for group, along in enumerate(("x", "z")):
        for k in range(n):
            off = (k - (n - 1) / 2) * spacing
            x_end = float(np.sqrt(max(r_in**2 - off**2, 1e-6))) + 0.03  # the ends sit inside the ring
            y0 = h / 2 - 0.004 + group * lath_h  # the second group lies over the first
            parts.append(S.solid(S.beam(_arch(x_end, v["dome_height"], y0, along, off), lath_w, lath_h), WOOD_LIGHT))
    for i in range(int(v["poles"])):  # roof poles from the ring outwards and down
        phi = 2 * np.pi * i / v["poles"]
        d = np.array([np.cos(phi), 0.0, np.sin(phi)])
        a = np.radians(v["pole_drop"])
        p0 = d * (R - 0.02)
        p1 = d * (R + v["pole_length"] * np.cos(a)) + np.array([0.0, -v["pole_length"] * np.sin(a), 0.0])
        parts.append(S.solid(S.tube(np.array([p0, p1]), 0.011, 6), WOOD_LIGHT))
    return S.join(parts)


register(Template(
    "shanyrak", "Шанырак",
    {
        "diameter": Param(1.8, 1.0, 3.2, "диаметр кольца", "м"),
        "ring_width": Param(0.10, 0.05, 0.25, "ширина кольца", "м"),
        "arches": Param(3, 2, 5, "дуг в каждой из двух групп (3 + 3 у настоящего шанырака)", integer=True),
        "dome_height": Param(0.30, 0.10, 0.70, "высота купола дуг", "м"),
        "poles": Param(72, 24, 120, "число уыков (жердей кровли)", integer=True),
        "pole_length": Param(0.9, 0.4, 1.8, "длина уыков", "м"),
        "pole_drop": Param(25, 10, 50, "наклон уыков вниз", "°"),
    },
    build,
    "Шанырак: толстое кольцо, две группы по три дуги крест-накрест, лучи-уыки.",
    ("шанырак", "шаңырақ", "shanyrak"),
))
