import math

import numpy as np
import pytest
import trimesh

from dronegen_server import meshing


def ball(n=40, r=14.0):
    g = np.mgrid[:n, :n, :n].astype(np.float64) - (n - 1) / 2
    return r - np.sqrt((g**2).sum(axis=0)), r


@pytest.mark.parametrize("backend", meshing.mc_backends())
def test_marching_cubes_winds_outwards(backend):
    vol, r = ball()
    v, f = meshing.marching_cubes(vol, 0.0, backend=backend)
    vol_mesh = meshing.signed_volume(v, f)
    assert vol_mesh == pytest.approx(4 / 3 * math.pi * r**3, rel=0.03)
    mesh = trimesh.Trimesh(v, f, process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent


def test_backends_agree():
    names = meshing.mc_backends()
    if len(names) < 2:
        pytest.skip("only one marching cubes library installed")
    vol, _ = ball(32, 11.0)
    a = meshing.signed_volume(*meshing.marching_cubes(vol, 0.0, backend=names[0]))
    b = meshing.signed_volume(*meshing.marching_cubes(vol, 0.0, backend=names[1]))
    assert a == pytest.approx(b, rel=0.01)


def test_colours_linear_in_glb_srgb_in_obj(tmp_path):
    box = trimesh.creation.box()
    srgb = np.tile([[0.5, 0.2, 1.0]], (len(box.vertices), 1))
    mesh = meshing.make_mesh(box.vertices, box.faces, srgb)
    meshing.export(mesh, tmp_path / "m.glb", tmp_path / "m.obj")
    glb = trimesh.load(tmp_path / "m.glb", force="mesh")
    lin = glb.visual.vertex_colors[0, :3] / 255.0
    assert lin == pytest.approx(meshing.srgb_to_linear(np.array([0.5, 0.2, 1.0])), abs=1 / 255)
    line = next(l for l in (tmp_path / "m.obj").read_text().splitlines() if l.startswith("v "))
    rgb = [float(x) for x in line.split()[4:7]]
    assert rgb == pytest.approx([0.5, 0.2, 1.0], abs=1 / 255 + 1e-6)


def test_normalize_and_decimate():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=3.0)
    colors = np.tile([[1.0, 0.0, 0.0]], (len(sphere.vertices), 1))
    mesh = meshing.normalize(meshing.make_mesh(sphere.vertices + 5.0, sphere.faces, colors))
    assert mesh.extents.max() == pytest.approx(1.0)
    assert np.allclose(mesh.bounds.mean(axis=0), 0.0, atol=1e-9)
    small, note = meshing.decimate(mesh, 500)
    if note:
        pytest.skip(note)
    assert len(small.faces) <= 520
    assert (small.visual.vertex_colors[:, 0] > 200).all()
