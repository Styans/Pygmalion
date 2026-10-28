"""Real models on the GPU (run with ``-m gpu``; skipped when weights or CUDA are missing).

    runtime\\venv\\Scripts\\python -m pytest server\\tests -m gpu -s
"""

import io
import time

import numpy as np
import pytest
import trimesh
from conftest import l_shape, silhouette_iou
from PIL import Image, ImageDraw

from dronegen_server.backends import Registry
from dronegen_server.config import Settings
from dronegen_server.jobs import JobManager

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def manager():
    from pathlib import Path

    home = Path(__file__).resolve().parents[2] / "runtime"
    settings = Settings.load(home, idle_unload_s=0)
    settings.apply_env()
    m = JobManager(settings, Registry(settings))
    m.start()
    yield m
    m.stop()


def run(manager, kind, backend, params, inputs=None, timeout=1800):
    b = manager.registry.get(kind, backend)
    ok, reason = b.check()
    if not ok:
        pytest.skip(f"{backend}: {reason}")
    job = manager.submit(kind, backend, params, inputs)
    t0 = time.time()
    while job.status not in ("done", "error", "cancelled"):
        assert time.time() - t0 < timeout
        time.sleep(0.2)
    assert job.status == "done", job.error
    return job


def flag_image():
    img = l_shape(400)
    ImageDraw.Draw(img).ellipse([250, 40, 340, 130], fill=(220, 30, 30))
    return img


def test_triposr_front_view_matches_picture(manager):
    img = flag_image()
    job = run(manager, "img2mesh", "triposr", {"mc_resolution": 192}, {"image": img})
    mesh = trimesh.load(manager.settings.outputs / job.files["mesh"], force="mesh")
    assert mesh.is_watertight and mesh.visual.kind == "vertex"
    score, mirrored = silhouette_iou(mesh, img)
    assert score > 0.85 and mirrored < score - 0.2, (score, mirrored)
    # the red disk is top-right in the picture: its vertices must be there in the mesh too
    rgb = mesh.visual.vertex_colors[:, :3].astype(int)
    red = mesh.vertices[(rgb[:, 0] > 120) & (rgb[:, 1] < 80) & (rgb[:, 2] < 80)]
    assert len(red) > 20 and red[:, 0].mean() > 0.1 and red[:, 1].mean() > 0.1
    print("TripoSR", job.metrics, job.result)


def test_zimage_turbo_picture(manager):
    params = {"prompt": "a red five-pointed star, single object, centered, plain white background, flat colors",
              "width": 512, "height": 512, "seed": 1}
    job = run(manager, "txt2img", "zimage_turbo", params)
    img = Image.open(manager.settings.outputs / job.files["image"])
    assert img.size == (512, 512)
    rgb = np.asarray(img.convert("RGB"), np.float32)
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3)])
    assert border.mean() > 200  # white background as asked
    centre = rgb[192:320, 192:320]
    assert (centre[..., 0] > centre[..., 2] + 40).mean() > 0.3  # something red in the middle
    print("Z-Image", job.metrics, job.result)


def test_zimage_then_triposr_switches_models(manager):
    """A txt2img model and TripoSR never sit in the GPU together: switching unloads the other."""
    zi = manager.registry.get("txt2img", "zimage_turbo")
    tr = manager.registry.get("img2mesh", "triposr")
    if not (zi.check()[0] and tr.check()[0]):
        pytest.skip("нужны обе модели")
    job = run(manager, "txt2img", "zimage_turbo", {"prompt": "golden eagle, wings spread, white background", "width": 512, "height": 512, "seed": 3})
    assert zi.loaded
    img = Image.open(io.BytesIO((manager.settings.outputs / job.files["image"]).read_bytes()))
    img.load()
    run(manager, "img2mesh", "triposr", {"mc_resolution": 128}, {"image": img})
    assert tr.loaded and not zi.loaded


def axes_object():
    """A known asymmetric object: a nose at the front (+Z), an arm on the right (+X), a chimney behind.

    The parts have their own colours: the front view must hint at the nose (a plain grey front
    gives the model no reason to believe the side view about a part sticking out at the front).
    """
    parts = []
    for size, centre, colour in (
        ((0.6, 1.0, 0.4), (0, 0, 0), (90, 120, 200)),
        ((0.22, 0.22, 0.3), (0, 0.2, 0.33), (220, 40, 40)),
        ((0.45, 0.15, 0.15), (0.5, 0.1, 0), (40, 200, 60)),
        ((0.14, 0.5, 0.14), (-0.18, 0.7, -0.1), (230, 210, 40)),
        ((0.2, 0.3, 0.2), (0.15, -0.6, 0.05), (160, 90, 40)),
    ):
        b = trimesh.creation.box(extents=size)
        b.apply_translation(centre)
        b.visual.vertex_colors = np.tile((*colour, 255), (len(b.vertices), 1)).astype(np.uint8)
        parts.append(b)
    return trimesh.util.concatenate(parts)


def card_of(mesh, direction, size=384):
    """What a camera at ``direction`` sees, cut out (RGBA)."""
    from dronegen_server import preview

    key = (255, 0, 255)
    rgb = np.asarray(preview.render(mesh, tuple(direction), size, background=key, margin=0.12))
    alpha = np.where(np.all(rgb == key, axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def chamfer(a, b, n=6000):
    from scipy.spatial import cKDTree

    def pts(m):
        p = m.sample(n)
        p -= (p.min(0) + p.max(0)) / 2
        return p / np.ptp(p, axis=0).max()

    pa, pb = pts(a), pts(b)
    return float(cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean())


def test_hunyuan_side_card_makes_the_shape_exact(manager):
    """Hunyuan's output is in glTF axes; a side card of unknown side is placed right and cuts the error."""
    from dronegen_server import cards

    gt = axes_object()
    front, side = card_of(gt, cards.CAMERA["front"]), card_of(gt, cards.CAMERA["left"])
    params = {"mc_resolution": 256, "colors": False}
    one = run(manager, "img2mesh", "hunyuan", params, {"image": front})
    mv = run(manager, "img2mesh", "hunyuan_mv", params, {"image": front, "views": {"front": front, "side": side}})
    assert "слева" in mv.result["side_note"], mv.result
    load = lambda job: trimesh.load(manager.settings.outputs / job.files["mesh"], force="mesh")  # noqa: E731
    m_one, m_mv = load(one), load(mv)
    turned = m_mv.copy()
    turned.vertices = m_mv.vertices * [-1, 1, -1]  # the same shape turned to face -Z
    c_one, c_mv, c_turned = chamfer(gt, m_one), chamfer(gt, m_mv), chamfer(gt, turned)
    print("chamfer: one picture", c_one, "cards", c_mv, "cards turned", c_turned, mv.metrics)
    assert c_mv < 0.05 and c_mv < c_one / 2 and c_turned > 2 * c_mv
