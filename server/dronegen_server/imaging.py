"""Image input/output and background removal.

Background removal methods:

* ``alpha``  - the image already has a transparent background;
* ``rembg``  - the U^2-Net salient-object model through onnxruntime on the CPU;
* ``simple`` - no neural network: the colour of the image border is the background, and
  every region of that colour connected to the border is cut away (plus large enclosed
  regions of exactly that colour, so a ring keeps its hole);
* ``auto``   - alpha if present, otherwise rembg, otherwise simple.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin, UnidentifiedImageError

from .errors import UserError

MAX_PIXELS = 4096 * 4096
BG_METHODS = ("auto", "rembg", "simple", "alpha")

_sessions: dict[str, object] = {}
_sessions_lock = threading.Lock()


def open_image(data: bytes) -> Image.Image:
    """Decode an uploaded file; EXIF rotation applied. Raises UserError for non-images."""
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise UserError(f"Слишком большая картинка ({img.width}×{img.height}); максимум 16 мегапикселей")
        img.load()
    except UserError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise UserError("Файл не удалось открыть как картинку (нужен PNG, JPEG или WebP)") from e
    return ImageOps.exif_transpose(img)


def to_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def has_alpha_matte(img: Image.Image) -> bool:
    """True when the alpha channel really separates a figure from a transparent background."""
    if img.mode not in ("RGBA", "LA") and not (img.mode == "P" and "transparency" in img.info):
        return False
    a = np.asarray(to_rgba(img))[..., 3]
    transparent = (a < 16).mean()
    opaque = (a > 240).mean()
    return transparent > 0.01 and opaque > 0.01


def rembg_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import rembg  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


def rembg_session(model: str = "u2net"):
    with _sessions_lock:
        if model not in _sessions:
            if "U2NET_HOME" not in os.environ:
                # never let rembg download into ~/.rembg: use our models folder even when the
                # caller (a script, a test) did not call Settings.apply_env()
                from .config import default_home

                os.environ["U2NET_HOME"] = str(default_home() / "models" / "u2net")
            import rembg

            _sessions[model] = rembg.new_session(model, providers=["CPUExecutionProvider"])
        return _sessions[model]


def remove_background(img: Image.Image, method: str = "auto") -> tuple[Image.Image, str]:
    """RGBA image with the background made transparent, and the method that was used."""
    if method not in BG_METHODS:
        raise UserError(f"Неизвестный способ удаления фона: {method}")
    if method in ("auto", "alpha") and has_alpha_matte(img):
        return to_rgba(img), "alpha"
    if method == "alpha":
        raise UserError("У картинки нет прозрачного фона — выберите другой способ удаления фона")
    if method == "rembg" or (method == "auto" and rembg_available()):
        import rembg

        out = rembg.remove(img.convert("RGB"), session=rembg_session())
        return to_rgba(out), "rembg"
    return simple_matte(img), "simple"


def border_color(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Median colour of the image border and how much the border varies (0..255)."""
    h, w = rgb.shape[:2]
    k = max(1, round(min(h, w) * 0.02))
    border = np.concatenate([rgb[:k].reshape(-1, 3), rgb[-k:].reshape(-1, 3), rgb[:, :k].reshape(-1, 3), rgb[:, -k:].reshape(-1, 3)])
    border = border.astype(np.float32)
    color = np.median(border, axis=0)
    spread = float(np.percentile(np.linalg.norm(border - color, axis=1), 90))
    return color, spread


def simple_matte(img: Image.Image, tolerance: float | None = None) -> Image.Image:
    """Background removal without a neural network (for plain, evenly coloured backgrounds)."""
    from scipy import ndimage

    rgba = np.asarray(to_rgba(img)).copy()
    rgb = rgba[..., :3].astype(np.float32)
    color, spread = border_color(rgb)
    tol = tolerance if tolerance is not None else max(24.0, 2.5 * spread)
    dist = np.linalg.norm(rgb - color, axis=-1)
    like_bg = dist < tol
    labels, n = ndimage.label(like_bg, structure=np.ones((3, 3), bool))
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, edge_labels[edge_labels > 0])
    # enclosed regions of exactly the background colour and of noticeable size are holes (a ring, a window)
    tight = dist < min(tol, 12.0)
    inner, m = ndimage.label(tight & ~background)
    if m:
        sizes = ndimage.sum(np.ones_like(inner), inner, index=np.arange(1, m + 1))
        figure_area = max(1, int((~background).sum()))
        holes = np.flatnonzero(sizes > 0.01 * figure_area) + 1
        background |= np.isin(inner, holes)
    fg = ~background
    fg = ndimage.binary_opening(fg, iterations=1)
    alpha = np.where(fg, 255, 0).astype(np.uint8)
    # a soft one-pixel edge so the colours at the rim are not the background's
    alpha = np.asarray(Image.fromarray(alpha).filter(_smooth_filter()))
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def _smooth_filter():
    from PIL import ImageFilter

    return ImageFilter.GaussianBlur(0.6)


def alpha_mask(rgba: Image.Image, threshold: int = 128) -> np.ndarray:
    return np.asarray(to_rgba(rgba))[..., 3] >= threshold


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(top, bottom, left, right), bottom/right exclusive. UserError if the mask is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise UserError("На картинке не найден объект: маска пустая. Попробуйте другой способ удаления фона")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def save_png(img: Image.Image, path: Path, meta: dict | None = None) -> Path:
    info = PngImagePlugin.PngInfo()
    if meta:
        info.add_text("dronegen", json.dumps(meta, ensure_ascii=False))
    img.save(path, format="PNG", pnginfo=info)
    return path


def read_meta(path: Path) -> dict:
    with Image.open(path) as img:
        text = img.info.get("dronegen")
    return json.loads(text) if text else {}
