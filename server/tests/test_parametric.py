import io

import numpy as np
import pytest
import trimesh
from conftest import wait_job

from dronegen_server import meshing, parametric
from dronegen_server.errors import UserError
from dronegen_server.parametric import dombra
from dronegen_server.parametric import shapes as S


@pytest.mark.parametrize("make", [
    lambda: S.lathe([(0.0, 0.0), (0.5, 0.2), (0.4, 0.8), (0.0, 1.0)], 32),
    lambda: S.tube([(0, 0, 0), (0, 0.5, 0.2), (0, 1, 0.1)], 0.05),
    lambda: S.beam([(0, 0, 0), (0.3, 0.5, 0), (0.6, 0, 0)], 0.1, 0.05),
    lambda: S.cylinder((0, 0, 0), (0, 1, 1), 0.1),
    lambda: S.sphere((0, 0, 0), 0.5),
    lambda: S.revolve_closed([(0.9, -0.03), (1.0, -0.03), (1.0, 0.03), (0.9, 0.03)], 48),
])
def test_primitives_are_closed_outward_solids(make):
    m = make()
    assert m.is_watertight and m.is_winding_consistent
    assert meshing.signed_volume(m.vertices, m.faces) > 0


def test_primitive_volumes():
    assert S.sphere((0, 0, 0), 1.0, 64).volume == pytest.approx(4 / 3 * np.pi, rel=0.01)
    assert S.cylinder((0, 0, 0), (0, 2, 0), 0.5, 64).volume == pytest.approx(np.pi * 0.25 * 2, rel=0.01)
    ring = S.revolve_closed([(0.9, -0.05), (1.0, -0.05), (1.0, 0.05), (0.9, 0.05)], 128)
    assert ring.volume == pytest.approx(0.1 * 0.1 * 2 * np.pi * 0.95, rel=0.01)


@pytest.mark.parametrize("key", ["dombra", "shanyrak", "yurt", "baiterek"])
def test_every_template_builds_closed_coloured_parts(key):
    mesh, values = parametric.build(key)
    assert mesh.is_watertight and meshing.signed_volume(mesh.vertices, mesh.faces) > 0
    assert mesh.visual.kind == "vertex" and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    assert set(values) == set(parametric.get(key).params)
    assert key in parametric.describe(key) and "Параметры" in parametric.describe(key)


def test_parameters_are_checked_and_applied():
    assert parametric.resolve("dombra", {"length": 1.2})["length"] == 1.2
    with pytest.raises(UserError, match="нет параметра «nonsense»"):
        parametric.resolve("dombra", {"nonsense": 1})
    with pytest.raises(UserError, match="от 0.7 до 1.3"):
        parametric.resolve("dombra", {"length": 5})
    with pytest.raises(UserError, match="числом"):
        parametric.resolve("dombra", {"length": "long"})
    with pytest.raises(UserError, match="Нет чертежа"):
        parametric.get("kobyz2")
    assert parametric.resolve("dombra", {"frets": 7.6})["frets"] == 8  # integers are rounded
    short, _ = parametric.build("dombra", {"length": 0.8})
    long, _ = parametric.build("dombra", {"length": 1.2})
    assert long.extents[1] / short.extents[1] == pytest.approx(1.5, rel=0.02)
    plain, _ = parametric.build("dombra", {"frets": 0, "soundhole": 0})
    full, _ = parametric.build("dombra", {"frets": 20})
    assert len(full.faces) > len(plain.faces)


def components_of_color(mesh, rgb_srgb, tol=0.03):
    """Connected parts whose vertex colour is (sRGB) ``rgb_srgb``."""
    lin = meshing.srgb_to_linear(np.array(rgb_srgb))
    out = []
    for part in mesh.split(only_watertight=False):
        c = part.visual.vertex_colors[:, :3].astype(float).mean(axis=0) / 255
        if np.all(np.abs(c - lin) < tol):
            out.append(part)
    return out


def test_dombra_has_exactly_two_pegs_two_strings_and_a_faceted_back():
    mesh, v = parametric.build("dombra")
    pegs = components_of_color(mesh, dombra.PEG)
    # each peg is a shaft plus a knob; one peg on each side of the head, at different heights
    sides = {}
    for part in pegs:
        sides.setdefault(int(np.sign(part.centroid[0])), []).append(part.centroid[1])
    assert sorted(sides) == [-1, 1] and all(len(ys) == 2 for ys in sides.values())
    assert abs(np.mean(sides[-1]) - np.mean(sides[1])) > 0.02
    strings = components_of_color(mesh, dombra.STRING)
    assert len(strings) == 2
    # the back is made of flat staves: 9 distinct face directions around the axis
    back = mesh.triangles_center[:, 2] < -0.01
    normals = mesh.face_normals[back & (mesh.triangles_center[:, 1] < 0.3)]
    angles = np.round(np.degrees(np.arctan2(normals[:, 2], normals[:, 0])), 0)
    assert 8 <= len(np.unique(angles[np.abs(normals[:, 1]) < 0.5])) <= 40  # a smooth back would give hundreds
    smooth, _ = parametric.build("dombra", {"back_facets": 0})
    assert len(smooth.faces) > len(mesh.faces)


def test_dombra_proportions():
    mesh, v = parametric.build("dombra")
    ext = mesh.extents
    assert ext[1] == pytest.approx(v["length"], rel=0.02)
    assert ext[0] == pytest.approx(v["body_width"], rel=0.05)  # the widest part is the body
    body = max(mesh.split(only_watertight=False), key=lambda p: p.volume)
    assert body.extents[2] / body.extents[0] == pytest.approx(v["body_depth"], rel=0.08)  # the depth of the back
    assert 0.3 <= body.extents[2] / body.extents[0] <= 0.5  # the range of a real dombra
    assert body.centroid[1] < 0.25 * v["length"]  # the body is at the bottom, the neck above


def test_shanyrak_has_the_crossed_arches_and_rays():
    mesh, v = parametric.build("shanyrak")
    parts = mesh.split(only_watertight=False)
    wood = components_of_color(mesh, parametric.shanyrak.WOOD_LIGHT)
    assert len(wood) == 2 * v["arches"] + v["poles"]
    two, _ = parametric.build("shanyrak", {"arches": 2, "poles": 30})
    assert len(components_of_color(two, parametric.shanyrak.WOOD_LIGHT)) == 4 + 30
    assert len(parts) > 70


def test_yurt_is_round_and_low():
    mesh, v = parametric.build("yurt")
    assert abs(mesh.extents[0] - v["diameter"]) < 0.05 and abs(mesh.extents[2] - v["diameter"]) < 0.1
    assert mesh.extents[1] < 0.6 * v["diameter"]
    plain, _ = parametric.build("yurt", {"door": 0, "bands": 0})
    assert len(plain.faces) < len(mesh.faces)


def test_baiterek_has_a_golden_sphere_with_the_real_ratio():
    mesh, v = parametric.build("baiterek", {"height": 2.0})
    gold = components_of_color(mesh, parametric.baiterek.GOLD)
    assert len(gold) == 1
    assert gold[0].extents[0] / mesh.extents[1] == pytest.approx(0.227, rel=0.03)  # 22 m of 97 m
    assert mesh.extents[1] == pytest.approx(2.0, rel=0.01)


def test_build_through_the_api(client):
    info = client.get("/templates").json()
    assert set(info) >= {"dombra", "shanyrak", "yurt", "baiterek"}
    assert info["dombra"]["params"]["frets"] == {"default": 14, "min": 0, "max": 24, "title": "число ладов", "unit": "", "integer": True}
    r = client.post("/build", json={"template": "dombra", "values": {"length": 1.1, "frets": 10}})
    assert r.status_code == 200, r.text
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["template"] == "dombra" and job["result"]["values"]["length"] == 1.1
    glb = client.get(f"/files/{job['files']['mesh']}").content
    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    assert mesh.visual.kind == "vertex" and mesh.extents.max() == pytest.approx(1.0, rel=0.01)  # normalised
    assert client.get(f"/files/{job['files']['preview']}").status_code == 200
    assert "build" in client.get("/health").json()["backends"]
    bad = client.post("/build", json={"template": "dombra", "values": {"length": 9}})
    assert bad.status_code == 400 and "от 0.7 до 1.3" in bad.json()["detail"]
    assert client.post("/build", json={"template": "tulpar"}).status_code == 400
