"""TripoSR adaptations that need no weights: the torchmcubes shim and the ViT key renaming."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dronegen_server import meshing  # noqa: E402
from dronegen_server.backends.triposr import VIT_PREFIX, remap_vit_keys, to_gltf_axes  # noqa: E402


def test_shim_matches_triposr_convention():
    """TripoSR: level = density - threshold (inside > 0), then v[..., [2, 1, 0]] / (res - 1)."""
    from dronegen_server import torchmcubes_shim

    n = 48
    g = torch.linspace(0, 1, n)
    x, y, z = torch.meshgrid(g, g, g, indexing="ij")
    # an off-centre ball, so a wrong axis order shows up as a shifted centre
    density = 0.3 - torch.sqrt((x - 0.35) ** 2 + (y - 0.5) ** 2 + (z - 0.6) ** 2)
    v, f = torchmcubes_shim.marching_cubes(density, 0.0)
    v = v[..., [2, 1, 0]] / (n - 1.0)
    assert v.dtype == torch.float32 and f.dtype == torch.int64
    centre = v.numpy().mean(axis=0)
    assert np.allclose(centre, [0.35, 0.5, 0.6], atol=0.01)
    assert meshing.signed_volume(v.numpy().astype(np.float64), f.numpy()) > 0  # outward


def test_vit_keys_old_to_new():
    transformers = pytest.importorskip("transformers")
    from transformers import ViTConfig, ViTModel

    model = ViTModel(ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=16))
    new_keys = {VIT_PREFIX + k for k in model.state_dict()}
    old = []
    for i in range(2):
        base = f"encoder.layer.{i}."
        old += [base + s for s in (
            "attention.attention.query.weight", "attention.attention.query.bias",
            "attention.attention.key.weight", "attention.attention.key.bias",
            "attention.attention.value.weight", "attention.attention.value.bias",
            "attention.output.dense.weight", "attention.output.dense.bias",
            "intermediate.dense.weight", "intermediate.dense.bias",
            "output.dense.weight", "output.dense.bias",
            "layernorm_before.weight", "layernorm_before.bias",
            "layernorm_after.weight", "layernorm_after.bias",
        )]
    old += ["embeddings.cls_token", "embeddings.position_embeddings", "embeddings.patch_embeddings.projection.weight",
            "embeddings.patch_embeddings.projection.bias", "layernorm.weight", "layernorm.bias",
            "pooler.dense.weight", "pooler.dense.bias"]
    state = {VIT_PREFIX + k: 0 for k in old}
    state["decoder.layers.0.weight"] = 0  # other parts of TripoSR stay as they are
    out = remap_vit_keys(state, new_keys | {"decoder.layers.0.weight"})
    assert set(out) == new_keys | {"decoder.layers.0.weight"}


def test_axes_are_a_rotation():
    m = np.eye(3)[[1, 2, 0]]
    assert np.isclose(np.linalg.det(m), 1.0)
    assert np.allclose(to_gltf_axes(np.array([[1.0, 2.0, 3.0]])), [[2.0, 3.0, 1.0]])
