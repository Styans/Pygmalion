import numpy as np
import pytest
from PIL import Image, ImageDraw

from dronegen_server import cards, preview
from dronegen_server.errors import UserError


def sheet(widths=(120, 50, 120), gap=90, height=300, inner_gap=0):
    """RGBA turnaround sheet: one figure per view on a transparent background."""
    w = sum(widths) + gap * (len(widths) + 1)
    img = Image.new("RGBA", (w, height + 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = gap
    for i, fw in enumerate(widths):
        color = [(200, 40, 40, 255), (40, 200, 40, 255), (40, 40, 200, 255)][i % 3]
        if inner_gap:  # a figure made of two parts with a narrow gap (an arm, a tail)
            d.rectangle([x, 40, x + fw // 2 - inner_gap // 2, 40 + height], fill=color)
            d.rectangle([x + fw // 2 + inner_gap // 2, 40, x + fw, 40 + height], fill=color)
        else:
            d.rectangle([x, 40, x + fw, 40 + height], fill=color)
        x += fw + gap
    return img


def colour(card):
    a = np.asarray(card)
    opaque = a[..., 3] > 128
    return tuple(int(c) for c in np.median(a[opaque][:, :3], axis=0))


def test_figures_left_to_right():
    (front, side, back), merged = cards.figures(sheet())
    assert [colour(c)[:3] for c in (front, side, back)] == [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
    widths = [imaging_width(c) for c in (front, side, back)]
    assert widths[1] < widths[0] and abs(widths[0] - widths[2]) <= 2 and not merged


def imaging_width(card):
    alpha = np.asarray(card)[..., 3] > 128
    cols = np.flatnonzero(alpha.any(axis=0))
    return cols[-1] - cols[0] + 1


def test_figures_keep_a_figure_with_a_narrow_gap_whole():
    parts, _ = cards.figures(sheet(inner_gap=6))
    assert len(parts) == 3
    assert imaging_width(parts[0]) >= 115  # both halves of the first figure


def test_figures_ignore_specks_and_an_empty_sheet_is_reported():
    img = sheet(widths=(120, 120))
    ImageDraw.Draw(img).rectangle([5, 5, 7, 7], fill=(0, 0, 0, 255))  # a speck far left
    assert len(cards.figures(img)[0]) == 2
    with pytest.raises(UserError, match="ни одной фигуры"):
        cards.figures(Image.new("RGBA", (600, 200), (0, 0, 0, 0)))


def test_square_centres_the_figure():
    card = cards.figures(sheet())[0][1]
    sq = cards.square(card, size=256, fill=0.8)
    assert sq.size == (256, 256)
    alpha = np.asarray(sq)[..., 3] > 128
    rows, cols = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
    assert rows[-1] - rows[0] + 1 == pytest.approx(0.8 * 256, abs=2)  # the long side fills 80 %
    assert abs((rows[0] + rows[-1]) / 2 - 127.5) < 2 and abs((cols[0] + cols[-1]) / 2 - 127.5) < 2


def test_mirror_flips_left_right():
    img = sheet(widths=(60,), gap=10)
    a = np.asarray(img)[..., 3]
    m = np.asarray(cards.mirror(img))[..., 3]
    assert np.array_equal(a[:, ::-1], m)


def test_camera_of_the_left_card_shows_the_front_on_the_picture_left():
    """«left» is the object's own left side: seen from +X, its front (+Z) is on the picture's left."""
    assert cards.CAMERA["left"] == (1.0, 0.0, 0.0) and cards.CAMERA["right"] == (-1.0, 0.0, 0.0)
    right, up, _ = preview._basis(cards.CAMERA["left"])
    assert np.allclose(up, (0, 1, 0)) and right @ np.array([0.0, 0.0, 1.0]) < 0
    right, _, _ = preview._basis(cards.CAMERA["right"])
    assert right @ np.array([0.0, 0.0, 1.0]) > 0


SKIN, SHIRT, SHELL, DARK = (220, 180, 140, 255), (200, 40, 40, 255), (40, 110, 50, 255), (60, 40, 30, 255)


def character(d, cx, view):
    """A standing figure as the image model draws its views: front and back share one outline (the back in other
    colours), the profile is narrower, its face on the picture's left and a shell on the right."""
    if view in ("front", "back"):
        head, body = (SKIN, SHIRT) if view == "front" else (DARK, SHELL)
        d.ellipse([cx - 45, 40, cx + 45, 130], fill=head)
        d.rectangle([cx - 70, 130, cx + 70, 330], fill=body)
        d.rectangle([cx - 100, 140, cx - 70, 280], fill=head)
        d.rectangle([cx + 70, 140, cx + 100, 280], fill=head)
    else:  # a profile facing left
        d.ellipse([cx - 60, 40, cx + 20, 130], fill=SKIN)
        d.polygon([(cx - 60, 70), (cx - 85, 95), (cx - 55, 105)], fill=SKIN)  # the nose
        d.rectangle([cx - 35, 130, cx + 30, 330], fill=SHIRT)
        d.ellipse([cx + 10, 140, cx + 80, 300], fill=SHELL)


def drawn_sheet(views):
    img = Image.new("RGBA", (260 * len(views) + 60, 380), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, view in enumerate(views):
        character(d, 160 + 260 * i, view)
    return img


def test_a_repeated_profile_is_not_taken_for_the_back_or_the_right_side():
    """Z-Image's usual sheet: front, the same profile twice, back. By their places the second profile would be the
    «back» and the back the «right» side; by what they show they are the side again and the back."""
    sheet_ = cards.read_sheet(drawn_sheet(["front", "side", "side", "back"]))
    assert sheet_.roles == ["front", "side", "side again", "back"]
    assert list(sheet_.views) == ["front", "side", "back"]
    assert colour(sheet_.views["front"]) == SHIRT[:3] and colour(sheet_.views["back"]) == SHELL[:3]
    assert "сбоку ×1" in sheet_.report()


def test_five_figures_give_each_view_once():
    sheet_ = cards.read_sheet(drawn_sheet(["front", "side", "back", "side", "back"]))
    assert sheet_.roles == ["front", "side", "back", "side again", "back again"]
    assert list(sheet_.views) == ["front", "side", "back"] and "5 фигур" in sheet_.report()


def test_a_front_drawn_again_is_no_back():
    sheet_ = cards.read_sheet(drawn_sheet(["front", "side", "front", "side"]))
    assert sheet_.roles == ["front", "side", "front again", "side again"]
    assert list(sheet_.views) == ["front", "side"] and "вида сзади нет" in sheet_.report()


def test_a_flat_object_seen_from_behind_and_edge_on():
    """A lightning bolt: drawn from the front again and again, and once mirrored - that is how its back looks.
    A flat board seen edge-on is a thin symmetric strip: still its side."""
    img = Image.new("RGBA", (1100, 380), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bolt = [(60, 0), (160, 0), (110, 110), (180, 110), (20, 330), (70, 170), (0, 170)]
    for i in range(4):
        x0 = 60 + 260 * i
        points = [(x0 + (180 - x if i == 3 else x), 20 + y) for x, y in bolt]
        d.polygon(points, fill=(30, 30, 30, 255))
    sheet_ = cards.read_sheet(img)
    assert sheet_.roles == ["front", "front again", "front again", "back"] and "вида сбоку нет" in sheet_.report()
    board = cards.read_sheet(sheet((200, 30, 200)))  # red front, green strip, blue back
    assert board.roles == ["front", "side", "back"] and colour(board.views["side"])[1] == 200


def test_compose_sheet_puts_the_views_in_one_row_with_one_scale():
    views = {v: Image.new("RGBA", (w, 200), (200, 40, 40, 255)) for v, w in (("front", 100), ("left", 60), ("back", 100))}
    out = cards.compose_sheet(views, height=400)
    assert out.height == 400 and out.width > 400
    a = np.asarray(out)[..., 0]
    columns = (np.abs(a.astype(int) - 236) > 20).any(axis=0)
    runs = cards._runs(columns, 3)
    assert len(runs) == 3 and runs[0][1] - runs[0][0] > runs[1][1] - runs[1][0]  # front wider than the side, same scale


def test_views_with_a_floor_shadow_and_touching_wings_still_split():
    """A soft shadow along the whole floor and wide figures that touch: the sheet is cut at the narrowest places."""
    h, w = 400, 1200
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(4):  # a body with wide thin wings; the wings of neighbours reach into each other
        cx = 150 + 300 * i
        d.rectangle([cx - 30, 120, cx + 30, 330], fill=(120, 60, 30, 255))
        d.polygon([(cx - 165, 140), (cx - 30, 160), (cx - 30, 200)], fill=(120, 60, 30, 255))
        d.polygon([(cx + 165, 140), (cx + 30, 160), (cx + 30, 200)], fill=(120, 60, 30, 255))
    d.rectangle([0, 340, w, 350], fill=(100, 100, 100, 255))  # the shadow band over the whole width
    figs, merged = cards.figures(img)
    assert len(figs) == 4 and not merged
    assert all(c.height == h and 150 < c.width < 450 for c in figs)


def test_one_wide_block_is_one_figure_not_cut_into_views():
    h, w = 400, 1200
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([100, 50, 1100, 350], fill=(120, 60, 30, 255))  # one wide block
    sheet_ = cards.read_sheet(img)
    assert sheet_.roles == ["front"] and list(sheet_.views) == ["front"]


def test_three_touching_figures_are_not_cut_into_four_views():
    """Z-Image sometimes draws three eagles instead of four: a wide one must not be cut in two and passed off as two views."""
    h, w = 400, 1200
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for cx in (200, 600, 1000):  # three figures with wings reaching each other
        d.rectangle([cx - 35, 120, cx + 35, 330], fill=(120, 60, 30, 255))
        d.polygon([(cx - 205, 140), (cx - 35, 160), (cx - 35, 200)], fill=(120, 60, 30, 255))
        d.polygon([(cx + 205, 140), (cx + 35, 160), (cx + 35, 200)], fill=(120, 60, 30, 255))
    figs, _ = cards.figures(img)
    assert len(figs) == 3
    assert "3 фигур" in cards.read_sheet(img).report()


def test_radially_symmetric_sheet_gives_front_side_back():
    """A Christmas tree, cone or vase looks the same from all four sides: read_sheet still gives front, side and back."""
    img = Image.new("RGBA", (1200, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(4):
        cx = 150 + 300 * i
        d.polygon([(cx, 40), (cx - 80, 340), (cx + 80, 340)], fill=(235, 240, 245, 255))
    sheet_ = cards.read_sheet(img)
    assert list(sheet_.views) == ["front", "side", "back"]
    assert sheet_.roles == ["front", "side", "back", "front again"]
    assert "симметричный предмет" in sheet_.report()


def test_single_figure_extracts_first_figure_from_sheet():
    s = sheet(widths=(120, 120, 120, 120), gap=80, height=300)
    fig, count = cards.single_figure(s)
    assert count == 4 and fig.width < s.width // 2
    assert colour(fig)[:3] == (200, 40, 40)
    one = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(one).rectangle([100, 50, 300, 350], fill=(200, 40, 40, 255))
    same, count_one = cards.single_figure(one)
    assert count_one == 1 and same is one

