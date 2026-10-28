"""SDXL base 1.0 (fp16, OpenRAIL++) as the fallback text-to-image model.

Slower than Z-Image-Turbo but independent of it, and it honours the negative prompt.
The fp16-fix VAE avoids black/NaN images in half precision.
"""

from __future__ import annotations

import logging

from .. import imaging, memory, weights
from . import diffusion
from .base import Txt2ImgBackend, register

log = logging.getLogger("dronegen.sdxl")

RAM_NEED_GB = 9.0
VRAM_OFFLOAD_GB = 7.0  # UNet 5.1 GB + activations at 1024²
VRAM_GROUP_GB = 2.0
GUIDANCE = 6.0


@register
class SDXL(Txt2ImgBackend):
    name = "sdxl"
    title = "SDXL (запасная модель)"
    needs_gpu = True
    heavy = True
    defaults = {**Txt2ImgBackend.defaults, "steps": 25, "width": 1024, "height": 1024}
    max_side = 1536

    def __init__(self, settings):
        super().__init__(settings)
        self.pipe = None
        self.mode = None

    def check(self):
        if not memory.cuda_ready():
            return False, "нужна видеокарта NVIDIA и PyTorch с CUDA (полный профиль установки)"
        missing = weights.missing(self.settings, "sdxl")
        return (False, weights.missing_message(missing)) if missing else (True, "")

    def load(self, ctx, force_mode: str | None = None):
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        memory.require_ram(RAM_NEED_GB, "SDXL", warn=ctx.warn)
        mode = force_mode or diffusion.requested_mode(ctx) or diffusion.pick_mode(memory.vram_free_gb(), VRAM_OFFLOAD_GB, VRAM_GROUP_GB)
        ctx.progress(0.02, "загрузка SDXL")
        vae = AutoencoderKL.from_pretrained(str(weights.folder(self.settings, "sdxl_vae")), dtype=torch.float16, local_files_only=True)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            str(weights.folder(self.settings, "sdxl")),
            vae=vae,
            dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        diffusion.apply_mode(self.pipe, mode)
        self.pipe.vae.enable_tiling()
        self.mode = mode
        if mode != "offload":
            ctx.warn(f"Мало свободной видеопамяти: SDXL работает в режиме «{diffusion.MODE_TITLES[mode]}» (медленнее).")

    def unload(self):
        self.pipe = None
        self.mode = None
        memory.free_gpu()
        self.loaded = False

    def run(self, ctx, params, inputs):
        import torch

        diffusion.ensure_mode(ctx, self)

        steps = params["steps"]

        def generate():
            generator = torch.Generator("cpu").manual_seed(params["seed"])
            with torch.inference_mode():
                return self.pipe(
                    prompt=params["prompt"],
                    negative_prompt=params.get("negative") or None,
                    width=params["width"],
                    height=params["height"],
                    num_inference_steps=steps,
                    guidance_scale=GUIDANCE,
                    generator=generator,
                    callback_on_step_end=diffusion.step_callback(ctx, steps, start=0.05),
                ).images[0]

        image = diffusion.run_with_fallback(ctx, self, generate)
        ctx.progress(0.97, "сохранение")
        meta = {k: params[k] for k in ("prompt", "negative", "width", "height", "steps", "seed")}
        meta.update({k: params[k] for k in ("subject", "style") if params.get(k)})
        imaging.save_png(image, ctx.output("image", "png"), {**meta, "backend": self.name, "memory_mode": self.mode})
        return {"seed": params["seed"], "prompt": params["prompt"], "width": params["width"], "height": params["height"],
                "memory_mode": self.mode}
