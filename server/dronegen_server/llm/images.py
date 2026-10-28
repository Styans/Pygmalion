"""Pictures the models look at: small JPEG copies named by their content.

A picture enters the chat history once, as a file in ``<home>/studio/llm``; every later request
reads the same bytes again (Claude binds its reasoning to the exact bytes of earlier turns).
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from pathlib import Path

from PIL import Image

MAX_SIDE = 1024
NAME = re.compile(r"^img_[0-9a-f]{20}\.jpg$")


def add(folder: Path, img: Image.Image, max_side: int = MAX_SIDE) -> str:
    """Store a picture for the models; returns its file name (the same picture -> the same name)."""
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
    if img.mode == "RGBA":  # cut-outs on a neutral grey, as the 3D models see them
        bg = Image.new("RGB", img.size, (200, 200, 200))
        bg.paste(img, mask=img.split()[3])
        img = bg
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    data = buf.getvalue()
    name = f"img_{hashlib.sha1(data).hexdigest()[:20]}.jpg"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.is_file():
        path.write_bytes(data)
    return name


def path(folder: Path, name: str) -> Path:
    if not NAME.match(name):
        raise ValueError(f"не картинка чата: {name}")
    return folder / name


def b64(folder: Path, name: str) -> str:
    return base64.standard_b64encode(path(folder, name).read_bytes()).decode("ascii")


def smaller_b64(folder: Path, name: str, max_side: int) -> str:
    """A smaller copy for the local model (fewer image tokens); deterministic."""
    img = Image.open(path(folder, name))
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")
