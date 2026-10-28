"""Shared parts of the diffusers backends: memory modes, progress callback, OOM fallback.

Memory modes, from fastest to leanest (chosen by free VRAM, stepped down on OOM):

* ``offload``    - each model goes to the GPU whole while it works (enable_model_cpu_offload);
* ``group``      - the denoiser streams blocks to the GPU, overlapped with compute (group offload);
* ``sequential`` - every layer is moved in on demand (slowest, needs the least VRAM).
"""

from __future__ import annotations

import logging

from .. import memory
from ..errors import OutOfMemory, VRAM_HINT, is_cuda_oom

log = logging.getLogger("dronegen.diffusion")

MODES = ("offload", "group", "sequential")
MODE_TITLES = {"offload": "обычный", "group": "экономный (поблочная подгрузка)", "sequential": "минимальный (медленно)"}


def pick_mode(free_gb: float, need_offload_gb: float, need_group_gb: float) -> str:
    if free_gb >= need_offload_gb:
        return "offload"
    if free_gb >= need_group_gb:
        return "group"
    return "sequential"


def apply_mode(pipe, mode: str) -> None:
    """Install the offloading hooks of ``mode`` on a pipeline that is still on the CPU."""
    import torch

    if mode == "offload":
        pipe.enable_model_cpu_offload()
    elif mode == "group":
        # layer by layer, the next layer is copied in on a side stream while the current one computes
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(mode)


def step_callback(ctx, steps: int, start: float = 0.15, end: float = 0.9):
    """diffusers callback_on_step_end: progress for the panel, and cancellation."""

    def cb(_pipe, i, _t, _kwargs):
        ctx.progress(start + (end - start) * (i + 1) / steps, f"шаг {i + 1} из {steps}")
        return {}

    return cb


def requested_mode(ctx) -> str | None:
    """A memory mode forced by the job (bench, advanced use); None = choose by free VRAM."""
    mode = ctx.job.params.get("memory_mode")
    return mode if mode in MODES else None


def ensure_mode(ctx, backend) -> None:
    """Reload ``backend`` when the job asks for another memory mode than the loaded one."""
    want = requested_mode(ctx)
    if want and want != backend.mode:
        backend.unload()
        memory.free_gpu()
        backend.load(ctx, force_mode=want)
        backend.loaded = True


def next_mode(mode: str) -> str | None:
    i = MODES.index(mode)
    return MODES[i + 1] if i + 1 < len(MODES) else None


def run_with_fallback(ctx, backend, generate):
    """Run ``generate()``; on CUDA out-of-memory reload ``backend`` in a leaner mode and retry."""
    while True:
        try:
            return generate()
        except Exception as e:  # noqa: BLE001
            if not is_cuda_oom(e):
                raise
            lower = next_mode(backend.mode)
            log.warning("%s: out of memory in mode %s -> %s", backend.name, backend.mode, lower)
            backend.unload()
            memory.free_gpu()
            if lower is None:
                raise OutOfMemory("Не хватило видеопамяти даже в минимальном режиме. " + VRAM_HINT) from e
            ctx.warn(f"Не хватило видеопамяти: включён режим «{MODE_TITLES[lower]}» — медленнее, но работает. "
                     "Освободите видеопамять, чтобы вернуть обычную скорость.")
            backend.load(ctx, force_mode=lower)
            backend.loaded = True
