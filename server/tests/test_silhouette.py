import numpy as np
import pytest
from conftest import l_shape, ring_image, silhouette_iou, two_color_bar
from PIL import Image, ImageDraw

from dronegen_server import imaging, meshing
from dronegen_server.backends.mock import draw_figure
from dronegen_server.backends.silhouette import inflate


def cut(img):
    return imaging.simple_matte(img)


def euler(mesh):
    return len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)


def test_ring_keeps_its_hole_and_is_closed():
    mesh = inflate(cut(ring_image()), resolution=96)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert meshing.signed_volume(mesh.vertices, mesh.faces) > 0
    assert euler(mesh) == 0  # a torus: genus 1


def test_disk_is_a_sphere_topologically():
    img = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(img).ellipse([50, 50, 250, 250], fill=(0, 0, 0))
    mesh = inflate(cut(img), resolution=80)
    assert mesh.is_watertight and euler(mesh) == 2


@pytest.mark.parametrize("make", [ring_image, l_shape, lambda: draw_figure(400, 400, 0), lambda: draw_figure(400, 400, 3)])
def test_front_view_matches_the_picture(make):
    img = make()
    mesh = inflate(cut(img), resolution=128)
    meshing.normalize(mesh)
    score, mirrored = silhouette_iou(mesh, img)
    assert score > 0.85, score
    if make is l_shape:
        assert mirrored < score - 0.2  # not mirrored left/right


def test_colours_come_from_the_front():
    img = two_color_bar()
    mesh = inflate(cut(img), resolution=96)
    rgb = mesh.visual.vertex_colors[:, :3].astype(float)
    x = mesh.vertices[:, 0]
    mid = (x.min() + x.max()) / 2
    left, right = rgb[x < mid - 5], rgb[x > mid + 5]
    assert (left[:, 0] > left[:, 2]).mean() > 0.95
    assert (right[:, 2] > right[:, 0]).mean() > 0.95


def test_depth_controls_thickness():
    img = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(img).rectangle([40, 40, 360, 360], fill=(0, 0, 0))
    thin = inflate(cut(img), resolution=64, depth=0.1)
    thick = inflate(cut(img), resolution=64, depth=0.6)
    t1 = thin.extents[2] / thin.extents[:2].max()
    t2 = thick.extents[2] / thick.extents[:2].max()
    assert 0.06 < t1 < 0.14 and t2 > 0.4


def test_thin_parts_stay_round_not_flat():
    """A thin limb gets about its own width in thickness, a wide body is capped."""
    img = Image.new("RGB", (500, 300), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 60, 240, 240], fill=(0, 0, 0))  # wide body
    d.rectangle([240, 140, 460, 160], fill=(0, 0, 0))  # thin arm, 20 px
    mesh = inflate(cut(img), resolution=160, depth=0.4)
    v = mesh.vertices
    arm = v[v[:, 0] > v[:, 0].min() + 0.75 * mesh.extents[0]]
    body = v[v[:, 0] < v[:, 0].min() + 0.3 * mesh.extents[0]]
    arm_t = arm[:, 2].max() - arm[:, 2].min()
    arm_w = arm[:, 1].max() - arm[:, 1].min()
    assert 0.6 < arm_t / arm_w < 1.4
    assert body[:, 2].max() > 2 * arm[:, 2].max()


def test_simple_matte_alpha_and_holes():
    rgba = imaging.simple_matte(ring_image())
    a = np.asarray(rgba)[..., 3]
    assert a[0, 0] == 0 and a[160, 160] == 0  # outside and the hole
    assert a[160, 40] == 255  # the ring itself


def test_alpha_input_is_used_as_is():
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).ellipse([40, 40, 160, 160], fill=(255, 255, 255, 255))  # white figure, no colour cue
    out, used = imaging.remove_background(rgba, "auto")
    assert used == "alpha"
    mesh = inflate(out, resolution=64)
    assert mesh.is_watertight


def test_empty_mask_is_a_user_error():
    from dronegen_server.errors import UserError

    with pytest.raises(UserError):
        inflate(Image.new("RGBA", (100, 100), (0, 0, 0, 0)), resolution=64)


def test_slab_is_a_plate_with_steep_walls():
    """«slab»: an exact cut-out of the outline with a flat face and walls, not a pillow."""
    img = l_shape()
    round_mesh = inflate(cut(img), resolution=128, depth=0.3)
    slab = inflate(cut(img), resolution=128, depth=0.3, profile="slab")
    for m in (round_mesh, slab):
        meshing.normalize(m)
        assert m.is_watertight
    score, _ = silhouette_iou(slab, img)
    assert score > 0.93  # the outline is followed more exactly than by the rounded volume

    # flat top: most of the surface facing the viewer lies in one plane (z == max)
    def flat_share(m):
        near = m.vertices[:, 2] > m.vertices[:, 2].max() - 0.02 * m.extents[2]
        return near.mean()

    assert flat_share(slab) > 2 * flat_share(round_mesh)
    # the walls are close to vertical: a thin ring around the outline carries little of the depth
    assert slab.volume > 1.3 * round_mesh.volume


def test_profile_is_checked(settings):
    from dronegen_server.backends.silhouette import Silhouette
    from dronegen_server.errors import UserError

    assert Silhouette(settings).validate({"profile": "slab"})["profile"] == "slab"
    with pytest.raises(UserError, match="profile"):
        Silhouette(settings).validate({"profile": "blob"})
