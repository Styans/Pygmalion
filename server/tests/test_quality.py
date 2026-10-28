import numpy as np
import pytest
import trimesh
from PIL import Image

from dronegen_server import meshing, quality


def sphere(noise=0.0):
    s = trimesh.creation.icosphere(subdivisions=4)
    v = s.vertices * (1 + noise * np.random.default_rng(0).standard_normal((len(s.vertices), 1)))
    return meshing.make_mesh(v, s.faces)


def test_depth_ratio_and_symmetry_of_a_box():
    b = trimesh.creation.box(extents=(1.0, 2.0, 0.4))
    assert quality.depth_ratio(b) == pytest.approx(0.4)
    assert quality.asymmetry(b) < 0.01


def test_bumpiness_tells_smooth_from_lumpy():
    smooth, lumpy = quality.bumpiness(sphere()), quality.bumpiness(sphere(0.03))
    assert smooth < 3 and lumpy > 3 * smooth


def test_view_iou_of_a_mesh_with_its_own_outline():
    b = trimesh.creation.box(extents=(1.0, 2.0, 0.4))
    a = np.zeros((220, 120, 4), np.uint8)
    a[10:210, 10:110] = (200, 50, 50, 255)  # the box from the front: 1 x 2
    img = Image.fromarray(a, "RGBA")
    assert quality.view_iou(b, "front", img) > 0.95
    assert quality.view_iou(b, (1.0, 0.0, 0.0), img) < 0.5  # from the side it is 0.4 x 2
    iou, mirrored = quality.front_iou(b, img)
    assert iou == pytest.approx(mirrored)
    assert set(quality.report(b, img)) >= {"depth_ratio", "bumpiness_deg", "asymmetry_pct", "front_iou"}
