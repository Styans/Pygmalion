import io
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from dronegen_server.config import Settings


@pytest.fixture()
def settings(tmp_path):
    s = Settings(home=tmp_path / "home", offline=True, idle_unload_s=0)
    s.ensure_dirs()
    s.apply_env()
    return s


@pytest.fixture()
def client(settings):
    from fastapi.testclient import TestClient

    from dronegen_server.app import create_app

    with TestClient(create_app(settings)) as c:
        yield c


def wait_job(client, job_id, timeout=120.0):
    t0 = time.time()
    while True:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        assert time.time() - t0 < timeout, f"job did not finish: {job}"
        time.sleep(0.05)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ring_image(size=320, color=(200, 30, 30)):
    """A ring (hole in the middle) on white."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    r1, r2 = size * 0.4, size * 0.2
    c = size / 2
    d.ellipse([c - r1, c - r1, c + r1, c + r1], fill=color)
    d.ellipse([c - r2, c - r2, c + r2, c + r2], fill="white")
    return img


def two_color_bar(w=360, h=160):
    """Left half red, right half blue: a wide bar to check colours and left/right."""
    img = Image.new("RGB", (w + 80, h + 80), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 40 + w // 2 - 1, 40 + h - 1], fill=(230, 20, 20))
    d.rectangle([40 + w // 2, 40, 40 + w - 1, 40 + h - 1], fill=(20, 40, 230))
    return img


def l_shape(size=300):
    """An asymmetric L: catches mirrored or rotated output."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 40, 120, 260], fill=(10, 150, 60))
    d.rectangle([60, 200, 250, 260], fill=(10, 150, 60))
    return img


def figure_mask(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), np.float32)
    return np.linalg.norm(rgb - 255.0, axis=-1) > 40


def front_projection(mesh, shape) -> np.ndarray:
    """Rasterise the mesh seen from the front (+Z, glTF axes) into a mask of ``shape``
    spanning the mesh's XY bounding box (row 0 = top)."""
    h, w = shape
    lo, hi = mesh.bounds[0][:2], mesh.bounds[1][:2]
    v = mesh.vertices[:, :2]
    cols = (v[:, 0] - lo[0]) / (hi[0] - lo[0]) * w
    rows = (hi[1] - v[:, 1]) / (hi[1] - lo[1]) * h
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for f in mesh.faces:
        d.polygon([(cols[i], rows[i]) for i in f], fill=255)
    return np.asarray(img) > 0


def crop_to(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def silhouette_iou(mesh, img: Image.Image, size=128) -> tuple[float, float]:
    """IoU of the mesh front view with the picture's figure, and with its mirror image."""
    want = crop_to(figure_mask(img))
    k = size / max(want.shape)
    shape = (max(1, round(want.shape[0] * k)), max(1, round(want.shape[1] * k)))
    want = np.asarray(Image.fromarray(want).resize(shape[::-1], Image.NEAREST))
    got = front_projection(mesh, shape)
    return iou(got, want), iou(got, want[:, ::-1])
