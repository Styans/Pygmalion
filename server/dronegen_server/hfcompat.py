"""Load checkpoints made with transformers 4 into transformers 5 models.

transformers 5 renamed the weights of its ViT and DINOv2 models (see its
``conversion_mapping.py``), but only applies the renaming in ``from_pretrained``.
TripoSR and Hunyuan3D build these models themselves and call ``load_state_dict`` on a
whole checkpoint, so the old keys are renamed here first. Keys the model already knows
are left alone, so the same code works with either transformers version.
"""

from __future__ import annotations

import re

_COMMON = [
    (re.compile(r"attention\.attention\.query"), "attention.q_proj"),
    (re.compile(r"attention\.attention\.key"), "attention.k_proj"),
    (re.compile(r"attention\.attention\.value"), "attention.v_proj"),
    (re.compile(r"attention\.output\.dense"), "attention.o_proj"),
]
VIT_RULES = [
    (re.compile(r"(^|\.)encoder\.layer\.(\d+)\."), r"\1layers.\2."),
    *_COMMON,
    (re.compile(r"intermediate\.dense"), "mlp.fc1"),
    (re.compile(r"(?<!attention\.)output\.dense"), "mlp.fc2"),
]
DINOV2_RULES = [*_COMMON, (re.compile(r"mlp\.weights_out"), "mlp.down_proj")]
_SWIGLU_IN = re.compile(r"mlp\.weights_in\.(weight|bias)$")


def _rename(key: str, rules) -> str:
    for pattern, repl in rules:
        key = pattern.sub(repl, key)
    return key


def remap(state: dict, model_keys, prefix: str, rules) -> dict:
    """Rename old-style keys under ``prefix`` that the model does not know."""
    model_keys = set(model_keys)
    out = {}
    for key, value in state.items():
        if key.startswith(prefix) and key not in model_keys:
            sub = key[len(prefix) :]
            m = _SWIGLU_IN.search(sub)
            if rules is DINOV2_RULES and m:
                # one fused SwiGLU input projection -> gate (activated half) and up
                gate, up = value.chunk(2, dim=0)
                base = prefix + _rename(sub[: m.start()], rules)
                out[f"{base}mlp.gate_proj.{m.group(1)}"] = gate
                out[f"{base}mlp.up_proj.{m.group(1)}"] = up
                continue
            key = prefix + _rename(sub, rules)
        out[key] = value
    return out


def remap_vit(state: dict, model_keys, prefix: str) -> dict:
    return remap(state, model_keys, prefix, VIT_RULES)


def remap_dinov2(state: dict, model_keys, prefix: str = "") -> dict:
    """DINOv2 anywhere in the state dict: the prefix is found from the keys themselves."""
    if prefix:
        return remap(state, model_keys, prefix, DINOV2_RULES)
    out = {}
    model_keys = set(model_keys)
    for key, value in state.items():
        if key in model_keys:
            out[key] = value
            continue
        i = key.find("encoder.layer.")
        if i < 0:
            out[key] = value
            continue
        out.update(remap({key: value}, model_keys, key[:i], DINOV2_RULES))
    return out
