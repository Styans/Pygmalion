"""Yurt (kiiz uy): a round felt dwelling.

Low cylindrical walls, a convex dome roof that ends in the shanyrak ring (the smoke hole stays
open), ropes around the roof, red and blue bands of ornament around the wall, a wooden door frame.
"""

from __future__ import annotations

import numpy as np

from . import shapes as S
from .registry import Param, Template, register

FELT = (0.93, 0.92, 0.87)
ROPE = (0.60, 0.50, 0.36)
RED = (0.72, 0.13, 0.10)
BLUE = (0.14, 0.31, 0.70)
GOLD = (0.92, 0.72, 0.22)
WOOD = (0.42, 0.24, 0.11)
DARK = (0.12, 0.08, 0.06)


def _dome(R: float, H: float, rc: float, roof: float, thick: float, samples: int = 24) -> list[tuple[float, float]]:
    """Closed (r, y) polygon of the roof shell: an outer quarter-ellipse and the same offset inwards."""
    s = np.linspace(0, 1, samples)
    outer = [(rc + (R - rc) * np.cos(a * np.pi / 2), H + roof * np.sin(a * np.pi / 2)) for a in s]
    inner = [(max(r - thick, rc * 0.6), y - thick * 0.9) for r, y in outer]
    return outer + inner[::-1]


def build(v: dict):
    R, H, roof = v["diameter"] / 2, v["wall_height"], v["roof_height"]
    rc = v["crown_diameter"] / 2
    parts = [
        S.solid(S.revolve_closed([(R - 0.06, 0.0), (R, 0.0), (R, H), (R - 0.06, H)], 96), FELT),
        S.solid(S.revolve_closed(_dome(R, H, rc, roof, 0.05), 96), FELT),
        # the crown ring over the smoke hole
        S.solid(S.revolve_closed([(rc - 0.07, H + roof - 0.04), (rc + 0.03, H + roof - 0.04), (rc + 0.03, H + roof + 0.05),
                                  (rc - 0.07, H + roof + 0.05)], 64), WOOD),
    ]
    for frac in (0.30, 0.62):  # ropes around the roof
        a = frac * np.pi / 2
        r, y = rc + (R - rc) * np.cos(a), H + roof * np.sin(a)
        parts.append(S.solid(S.revolve_closed([(r + 0.0, y - 0.018), (r + 0.045, y - 0.018), (r + 0.045, y + 0.018),
                                               (r + 0.0, y + 0.018)], 96), ROPE))
    if v["bands"] >= 0.5:
        for y, hh, color in ((0.84 * H, 0.13, RED), (0.60 * H, 0.09, BLUE), (0.92 * H, 0.03, GOLD), (0.30 * H, 0.03, GOLD)):
            parts.append(S.solid(S.revolve_closed([(R - 0.01, y - hh / 2), (R + 0.012, y - hh / 2), (R + 0.012, y + hh / 2),
                                                   (R - 0.01, y + hh / 2)], 96), color))
    if v["door"] >= 0.5:
        dw, dh, post = 0.95, 1.30, 0.07
        z = R + 0.015
        for x in (-dw / 2, dw / 2):
            parts.append(S.solid(S.box((x, dh / 2, z), (post, dh, 0.05)), WOOD))
        parts.append(S.solid(S.box((0, dh + post / 2, z), (dw + post, post, 0.05)), WOOD))
        parts.append(S.solid(S.box((0, dh / 2, z - 0.01), (dw - post, dh, 0.03)), DARK))
    return S.join(parts)


register(Template(
    "yurt", "Юрта",
    {
        "diameter": Param(6.0, 3.5, 10.0, "диаметр юрты", "м"),
        "wall_height": Param(1.6, 1.1, 2.2, "высота стен", "м"),
        "roof_height": Param(1.5, 0.8, 2.4, "высота купола над стенами", "м"),
        "crown_diameter": Param(1.3, 0.8, 2.2, "диаметр шанырака наверху", "м"),
        "bands": Param(1, 0, 1, "орнаментальные пояса (1 — есть)", integer=True),
        "door": Param(1, 0, 1, "дверь (1 — есть)", integer=True),
    },
    build,
    "Юрта: низкие круглые стены, выпуклый купол с открытым шаныраком, верёвки, красный и синий пояса, дверь.",
    ("юрта", "киіз үй", "yurt"),
))
