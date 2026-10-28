import pytest

from dronegen_server import styles


def test_render_style_adds_view_and_lighting():
    p = styles.build("a red star")
    assert p.startswith("a red star, front view, clean 3D product render")


def test_a_view_in_the_subject_is_kept():
    p = styles.build("Tulpar, the winged horse, side view.", "flat")
    assert p == f"Tulpar, the winged horse, side view, {styles.FLAT}"
    assert "front view" not in p


def test_sheet_wraps_the_subject():
    p = styles.build("a dombra", "sheet")
    assert "FRONT" in p and "LEFT SIDE" in p and "BACK" in p and "RIGHT SIDE" in p and "The object: a dombra." in p


def test_unknown_style():
    with pytest.raises(ValueError):
        styles.build("x", "watercolor")


@pytest.mark.parametrize("backend, style", [("hunyuan_mv", "sheet"), ("hunyuan", "render"), ("silhouette", "flat"), ("x", "render")])
def test_style_for_a_3d_method(backend, style):
    assert styles.for_mesh(backend) == style
