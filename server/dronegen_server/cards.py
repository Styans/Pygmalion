"""«Карточки» — views of one object (front / left / back / right) for multi-view 3D.

A turnaround sheet (one picture with the views side by side) is split by the empty
background between the views; every card is then cut out on its own.

Which view a drawn figure is, is read from the figures themselves, not from their place on the
sheet: Z-Image follows «front, left, back, right» only roughly. On 25 sheets it drew the front
first every time, but both profiles facing the same way (never a true right side), the front or
the back twice, and five figures instead of four. Taken by their places, such a sheet gives
Hunyuan3D a left profile as the «right» card (the object mirrored in depth) or a profile as the
«back». So a sheet gives front, side (of unknown side: ``hunyuan.place_side`` finds it) and back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import imaging
from .errors import UserError

VIEWS = ("front", "left", "back", "right")
VIEW_ORDER = ("front", "left", "side", "back", "right")  # how cards are laid out in a row
SHEET_FIGURES = 4  # the turnaround prompt asks for front, left, back and right
VIEW_WORDS = {"front": "front view", "left": "left side view in profile", "back": "back view",
              "right": "right side view in profile"}  # for drawing the views one by one
# Camera direction of each card in glTF axes. Like Hunyuan3D-2mv, a view is named after the
# side of the object it shows: a front-facing object has its own left on the picture's right,
# so «left» is seen from +X. Checked on a known asymmetric object (docs/BENCHMARKS.md).
CAMERA = {"front": (0.0, 0.0, 1.0), "left": (1.0, 0.0, 0.0), "back": (0.0, 0.0, -1.0), "right": (-1.0, 0.0, 0.0)}


def _runs(occupied: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """[start, end) runs of True, gaps shorter than ``min_gap`` bridged."""
    runs, start = [], None
    for i, v in enumerate(occupied):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(occupied)])
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def _without_shadow(rgba: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """The cut-out with the floor shadow and specks gone (a flat soft band across the whole sheet is part of the
    background, but the cut-out keeps it): only what is thicker than a feather stays. Returns the picture and its mask."""
    from scipy import ndimage

    rgba = imaging.to_rgba(rgba)
    alpha = np.asarray(rgba)[..., 3]
    thick = max(5, round(0.02 * alpha.shape[0]))
    core = ndimage.binary_opening(alpha > 128, structure=np.ones((thick, thick), bool))
    keep = ndimage.binary_dilation(core, structure=np.ones((thick, thick), bool), iterations=2)
    out = np.asarray(rgba).copy()
    out[..., 3] = np.where(keep, out[..., 3], 0)
    return Image.fromarray(out, "RGBA"), core


def _drop_neighbour_remnants(card: Image.Image) -> Image.Image:
    """After a cut through touching figures the card holds a piece of its neighbour (a wing tip) at the side edges:
    drop the small pieces that touch the left or right edge of the card."""
    from scipy import ndimage

    arr = np.asarray(card).copy()
    labels, n = ndimage.label(arr[..., 3] > 16)
    if n < 2:
        return card
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    big = sizes.max()
    for i in range(1, n + 1):
        if sizes[i - 1] < 0.25 * big and ((labels[:, 0] == i).any() or (labels[:, -1] == i).any()):
            arr[labels == i, 3] = 0
    return Image.fromarray(arr, "RGBA")


def _bodies(core: np.ndarray, count: int) -> list[tuple[int, int]]:
    """[x0, x1) of each figure's body, left to right: the thick part that stays when thin parts (wings, legs, a tail)
    are taken away. Pieces of one body above each other (a head and a trunk) count once.

    How thick a body is depends on the figure (a standing bird or wide-open wings, which are thick near the body):
    the thinnest erosion that finds ``count`` bodies is used. Thicker than 14 % of the height cuts one figure apart.
    """
    from scipy import ndimage

    found: list[tuple[int, int]] = []
    for frac in (0.08, 0.10, 0.12, 0.14):
        k = max(5, round(frac * core.shape[0]))
        thick = ndimage.binary_opening(core, structure=np.ones((k, k), bool))
        labels, n = ndimage.label(thick)
        if n == 0:
            break
        areas = ndimage.sum(thick, labels, index=range(1, n + 1))
        spans = sorted([sl[1].start, sl[1].stop] for i, sl in enumerate(ndimage.find_objects(labels))
                       if areas[i] >= 0.15 * areas.max())
        merged: list[list[int]] = []
        for x0, x1 in spans:
            if merged and x0 < merged[-1][1]:  # overlapping in x: one figure
                merged[-1][1] = max(merged[-1][1], x1)
            else:
                merged.append([x0, x1])
        bodies = [tuple(m) for m in merged]
        if len(bodies) == count:
            return bodies
        if len(bodies) > len(found):  # no erosion gives ``count``: the one that tells the most figures apart
            found = bodies
    return found


def figures(img: Image.Image) -> tuple[list[Image.Image], bool]:
    """Every figure of a turnaround sheet, left to right, background removed, each as tall as the sheet (so they keep
    one scale and one floor); and whether touching figures had to be cut where they grow into each other.

    Figures are found by the empty background between them. When they touch (wide wings, a long tail) the sheet is
    cut between their bodies, at its narrowest places: a wide figure is not cut in two.
    """
    from scipy import ndimage

    rgba, _ = imaging.remove_background(img, "auto")
    rgba, core = _without_shadow(rgba)
    h, w = core.shape
    column = core.any(axis=0)
    runs = [r for r in _runs(column, max(2, int(0.02 * w))) if r[1] - r[0] > 0.03 * w]
    if not runs:
        raise UserError("На листе видов не найдено ни одной фигуры: перегенерируйте лист")
    pad, view_width = int(0.02 * w), w / SHEET_FIGURES  # the width the prompt gives one view
    profile = ndimage.uniform_filter1d(core.sum(axis=0).astype(float), size=max(3, int(0.01 * w)))
    pieces, merged = [], False
    for x0, x1 in runs:
        # a run wider than one view may be touching figures: their bodies - what stays when everything thinner
        # than a body is taken away - tell them apart
        many = x1 - x0 > 1.5 * view_width
        bodies = _bodies(core[:, x0:x1], max(2, round((x1 - x0) / view_width))) if many else []
        if len(bodies) < 2:
            pieces.append(rgba.crop((max(0, x0 - pad), 0, min(w, x1 + pad), h)))
            continue
        cuts = [max(0, x0 - pad)]
        for (_, right), (left, _) in zip(bodies[:-1], bodies[1:]):
            lo, hi = sorted((x0 + right, x0 + left))
            cut = lo + int(np.argmin(profile[lo:hi + 1]))  # between the bodies, where the sheet is narrowest
            merged = merged or profile[cut] > 0.35 * profile.max()  # no narrow place: the views grow into each other
            cuts.append(cut)
        cuts.append(min(w, x1 + pad))
        pieces += [_drop_neighbour_remnants(rgba.crop((a, 0, b, h))) for a, b in zip(cuts[:-1], cuts[1:])]
    return [p for p in pieces if np.asarray(p)[..., 3].any()], merged


# Telling the drawn views apart (measured on 25 Z-Image sheets: turtles, eagles, dombras, lightning bolts).
# A repeat of a view matches it in shape (IoU 0.9+) and in colour (ΔE 12-15 on a repeated front, 3-7 on a
# repeated profile); a back has the front's outline (IoU 0.81-0.94) but other colours (ΔE 17-27). A profile is less
# like the front (IoU 0.26-0.81); where the two overlap, a back is symmetric (0.89-0.96) and a profile is not (< 0.7).
MATCH_HEIGHT = 160  # px: figures are compared at this sheet height
PROFILE_IOU = 0.85  # a profile matches the front (or its mirror) at most this well
BACK_IOU = 0.75  # a back matches the front (or its mirror) at least this well: a wider front again is no back
SYMMETRIC = 0.85  # IoU with its own mirror: between BACK_IOU and PROFILE_IOU a figure this symmetric is no profile
BACK_DE = 16.0  # CIE76 ΔE: a back differs from the front in colour at least this much; a repeated front less
MIRROR_GAIN = 0.15  # a figure that matches the front only mirrored (by this much more) is the back of a flat object
ROLE_WORDS = {"front": "спереди", "side": "сбоку", "back": "сзади"}


@dataclass
class Sheet:
    """What a turnaround sheet gives: ``views`` (front, side and back - those that are told apart; the side is of
    unknown side) and the role of every figure on it, left to right ("front", "side", "back" or "<view> again")."""

    views: dict[str, Image.Image]
    roles: list[str]
    merged: bool = False  # touching figures were cut where they grow into each other
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        """For the chat: which views were taken and what was left out."""
        text = "Карточки: " + ", ".join(ROLE_WORDS[v] for v in self.views)
        if self.notes:
            notes = "; ".join(self.notes)
            text += ". " + notes[0].upper() + notes[1:]
        return text


def _outline(card: Image.Image, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(mask, CIELAB colours) of a figure at ``scale``, cropped to its columns (the rows keep the floor)."""
    from skimage import color

    small = imaging.to_rgba(card)
    small = small.resize((max(1, round(small.width * scale)), max(1, round(small.height * scale))), Image.BILINEAR)
    a = np.asarray(small).astype(np.float32)
    cols = np.flatnonzero((a[..., 3] > 128).any(axis=0))
    if not len(cols):
        return np.zeros((a.shape[0], 1), bool), np.zeros((a.shape[0], 1, 3))
    a = a[:, cols[0]:cols[-1] + 1]
    return a[..., 3] > 128, color.rgb2lab(np.clip(a[..., :3] / 255.0, 0, 1))


def _mirrored(outline):
    mask, lab = outline
    return mask[:, ::-1], lab[:, ::-1]


def _match(a, b) -> tuple[float, float]:
    """How alike two figures are: (silhouette IoU, mean colour difference ΔE where they overlap) at the shift of ``b``
    that fits ``a`` best - the views are not drawn exactly under each other."""
    (ma, la), (mb, lb) = a, b
    h, wa, wb = ma.shape[0], ma.shape[1], mb.shape[1]
    reach = max(2, round(0.12 * max(wa, wb)))
    width = max(wa, wb) + 2 * reach
    xa = (width - wa) // 2
    big_a = np.zeros((h, width), bool)
    big_a[:, xa:xa + wa] = ma

    def placed(dx, dy, src, depth=None):
        out = np.zeros((h, width) if depth is None else (h, width, depth), src.dtype)
        x = (width - wb) // 2 + dx
        out[:, x:x + wb] = src
        return np.roll(out, dy, axis=0)

    best = (-1.0, 0, 0)
    step = max(1, round(0.01 * h))
    for dy in range(-2 * step, 2 * step + 1, step):
        for dx in range(-reach, reach + 1, 2):
            if not 0 <= (width - wb) // 2 + dx <= width - wb:
                continue
            big_b = placed(dx, dy, mb)
            iou = (big_a & big_b).sum() / max(1, (big_a | big_b).sum())
            if iou > best[0]:
                best = (float(iou), dx, dy)
    iou, dx, dy = best
    both = big_a & placed(dx, dy, mb)
    if not both.any():
        return iou, 100.0
    lab_a = np.zeros((h, width, 3))
    lab_a[:, xa:xa + wa] = la
    lab_b = placed(dx, dy, lb, 3)
    return iou, float(np.linalg.norm(lab_a[both] - lab_b[both], axis=1).mean())


def read_sheet(img: Image.Image) -> Sheet:
    """The views of a turnaround sheet by what the figures show, not by where they stand.

    The first figure is the front (the prompt puts it first, and Z-Image keeps to that). Of the others the side is
    the one least like the front, the back is the first that has the front's outline in other colours; repeats are
    left out. Which side the profile shows is not told: both profiles are drawn facing the same way.
    """
    figs, merged = figures(img)
    scale = MATCH_HEIGHT / figs[0].height
    outlines = [_outline(f, scale) for f in figs]
    front = outlines[0]
    profiles, backs, roles = [], [], ["front"] + ["front again"] * (len(figs) - 1)
    for i, x in enumerate(outlines[1:], 1):
        (iou, de), (iou_m, de_m) = _match(front, x), _match(front, _mirrored(x))
        symmetry = _match(x, _mirrored(x))[0]
        like_front = max(iou, iou_m)
        if like_front <= PROFILE_IOU and not (like_front >= BACK_IOU and symmetry >= SYMMETRIC):
            profiles.append((like_front + symmetry, i))
        elif min(de, de_m) >= BACK_DE or iou_m - iou >= MIRROR_GAIN:
            backs.append(i)
    views = {"front": figs[0]}
    symmetric_sheet = not profiles and not backs and len(figs) >= 2
    if profiles:
        side = min(profiles)[1]  # the most unlike the front: a true profile rather than three-quarters
        views["side"] = figs[side]
        for _, i in profiles:
            roles[i] = "side" if i == side else "side again"
    if backs:
        views["back"] = figs[backs[0]]
        for i in backs:
            roles[i] = "back" if i == backs[0] else "back again"
    if symmetric_sheet:
        # Radially symmetric / uniform-colour object (e.g. a Christmas tree, cone, sphere, vase, yurt):
        # all drawn views share the front's outline and colour, so take the next figures as side and back.
        views["side"] = figs[1]
        roles[1] = "side"
        if len(figs) >= 3:
            views["back"] = figs[2]
            roles[2] = "back"
    views = {v: views[v] for v in ("front", "side", "back") if v in views}

    notes = []
    if len(figs) != SHEET_FIGURES:
        notes.append(f"на листе {len(figs)} фигур(ы) вместо {SHEET_FIGURES}")
    if symmetric_sheet:
        notes.append("ракурсы одинаковы по силуэту и цвету (симметричный предмет)")
    again = [r.split()[0] for r in roles if r.endswith(" again")]
    if again and not symmetric_sheet:
        counts = ", ".join(f"{ROLE_WORDS[v]} ×{again.count(v)}" for v in ("front", "side", "back") if v in again)
        notes.append(f"повторы ракурсов не используются ({counts})")
    for view in ("side", "back"):
        if view not in views:
            notes.append(f"вида {ROLE_WORDS[view]} нет")
    if merged:
        notes.append("фигуры на листе слиплись: карточки вырезаны по самым узким местам")
    return Sheet(views, roles, merged, notes)


def single_figure(rgba: Image.Image) -> tuple[Image.Image, int]:
    """For single-view 3D backends: when given a turnaround sheet (a wide picture with several figures side by side),
    return only the first figure (front view) and the number of figures found, so one object is built instead of four."""
    if rgba.width < 1.4 * rgba.height:
        return rgba, 1
    try:
        figs, _ = figures(rgba)
    except UserError:
        return rgba, 1
    if len(figs) < 2:
        return rgba, 1
    return figs[0], len(figs)


def square(card: Image.Image, size: int = 1024, fill: float = 0.85) -> Image.Image:
    """Centre the figure on a square transparent canvas (what the 3D models expect)."""
    rgba = imaging.to_rgba(card)
    t, b, l, r = imaging.foreground_bbox(imaging.alpha_mask(rgba, 16))
    fig = rgba.crop((l, t, r, b))
    k = fill * size / max(fig.size)
    fig = fig.resize((max(1, round(fig.width * k)), max(1, round(fig.height * k))), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(fig, ((size - fig.width) // 2, (size - fig.height) // 2), fig)
    return out


def mirror(card: Image.Image) -> Image.Image:
    """A side view seen from the other side (valid for left/right symmetric objects)."""
    return card.transpose(Image.FLIP_LEFT_RIGHT)


def compose_sheet(views: dict[str, Image.Image], height: int = 640, background=(236, 236, 238)) -> Image.Image:
    """The views side by side on one picture (front, left, back, right), each cut out and equally scaled:
    the sheet that stands for the object in the chat. The 3D model still reads the separate cards."""
    order = [v for v in VIEW_ORDER if v in views]
    figs = []
    for name in order:
        rgba = imaging.to_rgba(views[name])
        t, b, l, r = imaging.foreground_bbox(imaging.alpha_mask(rgba, 16))
        figs.append(rgba.crop((l, t, r, b)))
    scale = 0.86 * height / max(f.height for f in figs)  # one scale for all: the views keep their proportions
    figs = [f.resize((max(1, round(f.width * scale)), max(1, round(f.height * scale))), Image.LANCZOS) for f in figs]
    pad = round(0.08 * height)
    width = sum(f.width for f in figs) + pad * (len(figs) + 1)
    sheet = Image.new("RGB", (width, height), background)
    x = pad
    for f in figs:
        sheet.paste(f, (x, height - f.height - round(0.06 * height)), f)
        x += f.width + pad
    return sheet
