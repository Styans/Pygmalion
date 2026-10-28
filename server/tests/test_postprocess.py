import numpy as np
import pytest
import trimesh
from PIL import Image

from dronegen_server import cards, meshing, postprocess, preview, quality


def box(extents, center=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return meshing.make_mesh(m.vertices, m.faces)


def lopsided():
    """A body with an arm on the +X side only, plus a floating speck."""
    body = box((0.6, 1.0, 0.4))
    arm = box((0.4, 0.15, 0.15), (0.5, 0.1, 0))
    speck = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
    speck.apply_translation((0, 0.9, 0))
    return trimesh.util.concatenate([body, arm, speck])


def test_drop_debris_keeps_big_parts_only():
    m = postprocess.drop_debris(lopsided())
    assert len(m.split(only_watertight=False)) == 2  # body and arm; the speck is gone
    assert m.bounds[1][1] < 0.6
    assert len(postprocess.drop_debris(lopsided(), keep_ratio=0.5).split(only_watertight=False)) == 1


def test_voxel_rebuild_is_watertight_and_keeps_the_size():
    m = postprocess.voxel_rebuild(box((0.6, 1.0, 0.4)), resolution=64)
    assert m.is_watertight
    assert np.allclose(m.extents, (0.6, 1.0, 0.4), atol=0.05)


def test_symmetry_mirrors_left_and_right():
    src = postprocess.drop_debris(lopsided())
    before = quality.asymmetry(postprocess.voxel_rebuild(src, resolution=80))
    sym = postprocess.voxel_rebuild(src, resolution=80, mirror_x=True)
    assert quality.asymmetry(sym) < before / 3
    # the mirror plane goes through the body, not the middle of body + arm: an arm on each side
    assert abs(sym.bounds[0][0] + sym.bounds[1][0]) < 0.03 and sym.bounds[1][0] > 0.68


def test_lowpoly_has_few_flat_faces():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    small, note = postprocess.lowpoly(meshing.make_mesh(sphere.vertices, sphere.faces), faces=300)
    assert note is None and len(small.faces) <= 320
    assert len(small.vertices) == 3 * len(small.faces)  # unshared vertices: faceted shading


def test_set_depth():
    m = postprocess.set_depth(box((1.0, 2.0, 0.8)), 0.3)
    assert m.extents[2] / m.extents[0] == pytest.approx(0.3)


def card(color_left, color_right, size=(200, 300)):
    """A card whose figure is a rectangle: one colour on the picture's left half, another on the right."""
    a = np.zeros((size[1], size[0], 4), np.uint8)
    a[20:-20, 20 : size[0] // 2] = (*color_left, 255)
    a[20:-20, size[0] // 2 : -20] = (*color_right, 255)
    return Image.fromarray(a, "RGBA")


def srgb_of(mesh, mask):
    lin = mesh.visual.vertex_colors[mask, :3].astype(float) / 255
    return meshing.linear_to_srgb(lin).mean(axis=0) * 255


def test_colours_come_from_the_card_the_surface_faces():
    m = box((1.0, 1.5, 1.0))
    m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.1), process=False)
    red, blue, green = (230, 20, 20), (20, 20, 230), (20, 200, 20)
    postprocess.colorize_from_views(m, {"front": card(red, blue), "left": card(green, green)})
    n = m.vertex_normals
    front_left = (n[:, 2] > 0.9) & (m.vertices[:, 0] < -0.1)
    front_right = (n[:, 2] > 0.9) & (m.vertices[:, 0] > 0.1)
    plus_x = n[:, 0] > 0.9  # the «left» card is seen from +X
    assert np.argmax(srgb_of(m, front_left)) == 0  # picture left = -X is red
    assert np.argmax(srgb_of(m, front_right)) == 2
    assert np.argmax(srgb_of(m, plus_x)) == 1


def test_a_three_quarter_card_is_projected_from_where_it_was_drawn():
    """A «front» picture drawn in three-quarter view from above (as image models like to draw a car): projected
    straight from the front, its colours slide along the body; from its real direction they land in place."""
    body = trimesh.creation.box(extents=(0.5, 0.25, 1.0))  # long along Z, its nose at +Z
    cabin = trimesh.creation.box(extents=(0.4, 0.2, 0.45))
    cabin.apply_translation((0, 0.225, -0.15))  # on the back half: a car has a top and a bottom, a box does not
    car = trimesh.util.concatenate([body, cabin])
    v, f = trimesh.remesh.subdivide_to_size(car.vertices, car.faces, 0.04)
    nose = v[:, 2] > 0
    truth = meshing.make_mesh(v, f, np.where(nose[:, None], (0.9, 0.1, 0.1), (0.1, 0.1, 0.9)))
    d = postprocess._direction(-30, 15)
    img = preview.render(truth, d, 256)
    rgb = np.asarray(img)
    alpha = np.where((rgb != preview.BACKGROUND).any(axis=-1), 255, 0).astype(np.uint8)
    card = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

    found, score, at_own = postprocess.card_camera(truth, "front", card)
    assert found @ d > 0.97 and score > at_own + 0.1

    m = trimesh.Trimesh(v, f, process=False)
    postprocess.colorize_from_views(m, {"front": card})
    n = m.vertex_normals
    nose_face = n[:, 2] > 0.9
    side = (n[:, 0] < -0.9) & (m.vertices[:, 0] < -0.24) & (np.abs(m.vertices[:, 2]) > 0.15)  # -X: the camera sees it
    got_red = srgb_of(m, side & nose)[0] > srgb_of(m, side & nose)[2]
    got_blue = srgb_of(m, side & ~nose)[2] > srgb_of(m, side & ~nose)[0]
    assert np.argmax(srgb_of(m, nose_face)) == 0  # the whole nose is red, not half red, half blue
    assert got_red and got_blue  # along the side the colour changes where the nose ends


def test_apply_runs_every_step():
    src = lopsided()
    red = card((230, 20, 20), (230, 20, 20))
    m = postprocess.apply(src, {"style": "lowpoly", "lowpoly_faces": 200, "symmetry": True, "depth_ratio": 0.5,
                                "voxel_resolution": 64}, {"front": red})
    assert len(m.faces) <= 220
    assert m.extents[2] / m.extents[0] == pytest.approx(0.5, abs=0.05)
    assert m.visual.kind == "vertex"


def test_preview_and_silhouette_agree_with_the_axes():
    m = postprocess.drop_debris(lopsided())
    img = preview.render(m, "front", 128)
    assert img.size == (128, 128)
    sil = preview.silhouette(m, "front", (100, 100))
    assert sil[:, 80:].sum() > 0 and sil[:, 80:].sum() < sil[:, :50].sum()  # the arm (+X) is on the right
    side = preview.silhouette(m, cards.CAMERA["left"], (100, 60))
    assert side.mean() > 0.5
    assert preview.sheet(m, size=64).size == (192, 64)


def test_a_part_behind_another_gets_no_colour_from_that_camera():
    """A small ball hidden behind a big one is not seen from the front: it must not take the front card's colour."""
    big = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    small = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
    small.apply_translation((0, 0, -1.6))  # behind the big ball, in line with the camera
    mesh = trimesh.util.concatenate([big, small])
    seen = postprocess._visible(mesh, (0.0, 0.0, 1.0))
    hidden = np.asarray(mesh.vertices)[:, 2] < -1.2
    assert hidden.any() and not seen[hidden].any()
    assert seen[np.asarray(mesh.vertices)[:, 2] > 0.8].all()  # the front of the big ball is seen


def test_front_only_card_still_gives_every_side_a_colour():
    """One front card: the back is its mirror image, the sides are filled from the neighbours: no stripes, no holes."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    red = Image.new("RGBA", (200, 200), (210, 30, 30, 255))
    postprocess.colorize_from_views(mesh, {"front": red})
    rgb = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(int)
    assert (rgb[:, 0] > 120).all() and (rgb[:, 1] < 60).all() and (rgb[:, 2] < 60).all()


def test_one_front_card_does_not_paint_a_face_on_the_back():
    """With a front card only, the back is guessed from it seen through the object: its colours, not its details.
    Sharp, it put the eagle's eyes and beak on the back of its head and the spots of its chest on its back."""
    mesh = trimesh.creation.uv_sphere(radius=1.0, count=[64, 64])  # its poles: the middle of the front and the back
    card = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    from PIL import ImageDraw

    d = ImageDraw.Draw(card)
    d.ellipse([0, 0, 299, 299], fill=(210, 30, 30, 255))  # a red head ...
    d.ellipse([144, 144, 156, 156], fill=(20, 20, 20, 255))  # ... with a small black eye in the middle
    postprocess.colorize_from_views(mesh, {"front": card})
    v, n = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    middle = np.hypot(v[:, 0], v[:, 1]) < 0.03
    eye, back_of_head = middle & (n[:, 2] > 0.9), middle & (n[:, 2] < -0.9)
    assert eye.any() and back_of_head.any()
    assert srgb_of(mesh, eye).max() < 80  # the front keeps its detail
    back = srgb_of(mesh, back_of_head)
    assert back[0] > 150 and back[0] > back[1] + 80  # the back is the red of the head, no dark eye


def test_the_light_rim_of_a_cutout_is_not_used_as_colour():
    """The edge of a cut-out is a blend with the old background: the colour is taken from inside the figure."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    card = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    from PIL import ImageDraw

    d = ImageDraw.Draw(card)
    d.ellipse([10, 10, 190, 190], fill=(250, 250, 250, 255))  # a white halo ...
    d.ellipse([16, 16, 184, 184], fill=(30, 30, 200, 255))  # ... around a blue figure
    postprocess.colorize_from_views(mesh, {"front": card, "back": card})
    rgb = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(int)
    assert (rgb[:, 2] > rgb[:, 0] + 60).mean() > 0.97  # blue, not whitish
