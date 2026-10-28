"""Hunyuan3D backends without the GPU: settings, cards, the side-card choice, DINOv2 key renaming."""

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from dronegen_server import cards, hfcompat, meshing
from dronegen_server.backends.hunyuan import Hunyuan, HunyuanMV
from dronegen_server.errors import UserError


class Ctx:
    def __init__(self):
        self.warnings, self.stages = [], []

    def progress(self, fraction, stage=None):
        self.stages.append(stage)

    def warn(self, message):
        self.warnings.append(message)


def rgba_rect(w=120, h=300, color=(180, 120, 60)):
    img = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([20, 20, 20 + w, 20 + h], fill=(*color, 255))
    return img


def test_defaults_and_bad_values(settings):
    p = Hunyuan(settings).validate({})
    assert (p["steps"], p["mc_resolution"], p["style"], p["symmetry"], p["colors"]) == (5, 384, "as_is", False, True)
    assert HunyuanMV(settings).validate({})["side"] == "auto"
    for bad in ({"style": "watercolor"}, {"steps": 0}, {"depth_ratio": 5}, {"mc_resolution": 1024}):
        with pytest.raises(UserError):
            Hunyuan(settings).validate(bad)
    with pytest.raises(UserError, match="side"):
        HunyuanMV(settings).validate({"side": "top"})


def test_sheet_becomes_front_side_back(settings):
    mv = HunyuanMV(settings)
    sheet = Image.new("RGBA", (900, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for x0, w, colour in ((80, 150, (150, 100, 50)), (380, 60, (150, 100, 50)), (600, 150, (40, 90, 160))):
        d.rectangle([x0, 40, x0 + w, 360], fill=(*colour, 255))
    ctx = Ctx()
    views, used = mv.views(ctx, mv.validate({}), {"image": sheet})
    assert list(views) == ["front", "side", "back"] and used == "alpha"
    assert views["side"].width < views["front"].width
    assert any("3 фигур" in w for w in ctx.warnings)  # what the sheet lacks is told


def test_sheet_with_repeated_views_gives_each_view_once(settings):
    """Front, the same profile twice, the front again: not four views but a front and a side."""
    mv = HunyuanMV(settings)
    sheet = Image.new("RGBA", (1200, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for x0, w in ((60, 150), (360, 80), (640, 80), (940, 150)):
        d.rectangle([x0, 40, x0 + w, 360], fill=(150, 100, 50, 255))
    views, _ = mv.views(Ctx(), mv.validate({}), {"image": sheet})
    assert list(views) == ["front", "side"]


def test_explicit_cards_and_their_errors(settings):
    mv = HunyuanMV(settings)
    p = mv.validate({})
    views, _ = mv.views(Ctx(), p, {"views": {"front": rgba_rect(), "left": rgba_rect(50)}})
    assert set(views) == {"front", "left"}
    for given, text in (
        ({"left": rgba_rect(), "back": rgba_rect()}, "спереди"),
        ({"front": rgba_rect(), "top": rgba_rect()}, "Неизвестные виды: top"),
        ({"front": rgba_rect(), "side": rgba_rect(), "left": rgba_rect()}, "side"),
        ({"front": rgba_rect()}, "хотя бы 2"),
    ):
        with pytest.raises(UserError, match=text):
            mv.views(Ctx(), p, {"views": given})


def test_single_view_hunyuan_takes_first_figure_from_sheet(settings):
    """A 4-view sheet passed to single-view Hunyuan must build one object (the first figure), not four in a row."""
    hy = Hunyuan(settings)
    sheet = Image.new("RGBA", (1200, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i in range(4):
        x0 = 60 + 280 * i
        d.rectangle([x0, 40, x0 + 140, 360], fill=(220, 220, 220, 255))
    ctx = Ctx()
    views, _ = hy.views(ctx, hy.validate({}), {"image": sheet})
    assert list(views) == ["front"]
    assert views["front"].width < sheet.width // 2
    assert any("лист из 4 видов" in w for w in ctx.warnings)


def test_hunyuan_mv_accepts_radially_symmetric_sheet_and_single_figure_image(settings):
    mv = HunyuanMV(settings)
    p = mv.validate({})
    sheet = Image.new("RGBA", (1200, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i in range(4):
        cx = 150 + 300 * i
        d.polygon([(cx, 40), (cx - 80, 340), (cx + 80, 340)], fill=(235, 240, 245, 255))
    views, _ = mv.views(Ctx(), p, {"image": sheet})
    assert list(views) == ["front", "side", "back"]
    assert all(card.width < sheet.width // 2 for card in views.values())

    ctx = Ctx()
    one_views, _ = mv.views(ctx, p, {"image": rgba_rect(120, 300)})
    assert list(one_views) == ["front"]


def l_card():
    """An L seen from the side: a tall bar on the picture's left, a foot to the right."""
    img = Image.new("RGBA", (240, 320), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 80, 300], fill=(90, 90, 200, 255))
    d.rectangle([20, 240, 220, 300], fill=(90, 90, 200, 255))
    return img


def l_mesh():
    """A mesh that shows ``l_card`` from +X: the picture's left is +Z there."""
    bar = trimesh.creation.box(extents=(0.2, 1.0, 0.2))
    bar.apply_translation((0, 0, 0.3))
    foot = trimesh.creation.box(extents=(0.2, 0.2, 0.8))
    foot.apply_translation((0, -0.4, 0.0))
    m = trimesh.util.concatenate([bar, foot])
    return meshing.make_mesh(m.vertices, m.faces)


@pytest.mark.parametrize("mirrored, expected", [(False, "left"), (True, "right")])
def test_side_card_goes_where_its_silhouette_fits(settings, monkeypatch, mirrored, expected):
    mv = HunyuanMV(settings)
    side = cards.mirror(l_card()) if mirrored else l_card()
    tried = []
    monkeypatch.setattr(mv, "diffuse", lambda ctx, params, views, span=None: tried.append(sorted(views)) or dict(views))
    monkeypatch.setattr(mv, "decode", lambda ctx, latents, res, chunks: l_mesh())
    ctx = Ctx()
    views, latents, note = mv.place_side(ctx, mv.validate({}), {"front": front_of_l(), "side": side})
    assert tried == [["front", "left"], ["front", "right"]]
    assert set(views) == {"front", expected} and views[expected] is side
    assert latents == views and ("слева" if expected == "left" else "справа") in note
    assert not ctx.warnings  # a clear difference


def test_unsure_side_choice_is_reported(settings, monkeypatch):
    mv = HunyuanMV(settings)
    monkeypatch.setattr(mv, "diffuse", lambda ctx, params, views, span=None: dict(views))
    monkeypatch.setattr(mv, "decode", lambda *a: meshing.make_mesh(*_box(0.4, 1.0, 0.4)))
    ctx = Ctx()
    views, _, _ = mv.place_side(ctx, mv.validate({}), {"front": rgba_rect(), "side": rgba_rect()})
    assert set(views) in ({"front", "left"}, {"front", "right"})
    assert ctx.warnings and "side: left или right" in ctx.warnings[0]


def _box(*extents):
    b = trimesh.creation.box(extents=extents)
    return b.vertices, b.faces


def front_of_l():
    """The L mesh from the front (+Z): a 0.2 x 1.0 bar, with the foot hidden behind it."""
    return rgba_rect(40, 200)


def test_side_card_with_a_known_side_needs_no_trials(settings, monkeypatch):
    mv = HunyuanMV(settings)
    monkeypatch.setattr(mv, "diffuse", lambda *a, **k: pytest.fail("no trial runs expected"))
    views, latents, note = mv.place_side(Ctx(), mv.validate({"side": "right"}), {"front": rgba_rect(), "side": l_card()})
    assert set(views) == {"front", "right"} and latents is None and note == ""


def test_dinov2_old_keys_load_into_transformers5_model():
    torch = pytest.importorskip("torch")
    from transformers import Dinov2Config, Dinov2Model

    model = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                                     image_size=28, patch_size=14, use_swiglu_ffn=True))
    new = model.state_dict()
    old_names = {"attention.q_proj": "attention.attention.query", "attention.k_proj": "attention.attention.key",
                 "attention.v_proj": "attention.attention.value", "attention.o_proj": "attention.output.dense",
                 "mlp.down_proj": "mlp.weights_out"}
    prefix = "main_image_encoder.model."
    old = {}
    for key, value in new.items():
        if ".mlp.gate_proj." in key or ".mlp.up_proj." in key:
            continue
        for a, b in old_names.items():
            key = key.replace(a, b)
        old[prefix + key] = torch.randn_like(value)
    for kind in ("weight", "bias"):
        gate, up = new[f"encoder.layer.0.mlp.gate_proj.{kind}"], new[f"encoder.layer.0.mlp.up_proj.{kind}"]
        old[f"{prefix}encoder.layer.0.mlp.weights_in.{kind}"] = torch.cat([torch.randn_like(gate), torch.randn_like(up)])
    out = hfcompat.remap_dinov2(old, {prefix + k for k in new})
    assert set(out) == {prefix + k for k in new}
    fused = old[f"{prefix}encoder.layer.0.mlp.weights_in.weight"]
    assert torch.equal(out[f"{prefix}encoder.layer.0.mlp.gate_proj.weight"], fused[: len(fused) // 2])
    model.load_state_dict({k[len(prefix):]: v for k, v in out.items()}, strict=True)
    assert np.allclose(model.state_dict()["encoder.layer.0.attention.q_proj.weight"].numpy(),
                       old[prefix + "encoder.layer.0.attention.attention.query.weight"].numpy())
