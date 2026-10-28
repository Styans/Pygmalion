"""Z-Image-Turbo (Tongyi-MAI, 6B, Apache-2.0) through diffusers' ZImagePipeline, tuned for ~8 GB GPUs.

* The transformer comes from a GGUF file (Q4_K_M, 4.6 GB instead of 24.6 GB of fp32 weights)
  and computes in bf16.
* The text encoder (Qwen3-4B, ~8 GB in bf16) never touches the GPU and is not kept in RAM:
  for a new prompt it is loaded (3–5 s), encodes on the CPU (<1 s) and is freed again; the
  embeddings are cached, so new seeds of the same prompt skip it. ``zimage_keep_text_encoder``
  keeps it loaded instead (faster, ~8 GB more RAM). The prompt is not padded to 512 tokens.
* Turbo is distilled: 9 steps (8 transformer passes) and no CFG, so the negative prompt has no
  effect.
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict

from .. import imaging, memory, weights
from ..errors import UserError
from . import diffusion
from .base import Txt2ImgBackend, register

log = logging.getLogger("dronegen.zimage")

RAM_LOAD_GB = 7.0  # GGUF transformer kept on the CPU between jobs + VAE + runtime
RAM_ENCODER_GB = 8.5  # Qwen3-4B in bf16 while encoding
VRAM_OFFLOAD_GB = 6.5  # transformer whole on the GPU (4.7) + activations up to 1 MP, with a margin
VRAM_GROUP_GB = 2.0
CACHE_PROMPTS = 32


@register
class ZImageTurbo(Txt2ImgBackend):
    name = "zimage_turbo"
    title = "Z-Image-Turbo"
    needs_gpu = True
    heavy = True
    defaults = {**Txt2ImgBackend.defaults, "steps": 9}
    max_side = 1920

    def __init__(self, settings):
        super().__init__(settings)
        self.pipe = None
        self.tokenizer = None
        self.text_encoder = None
        self.mode = None
        self.embeds: OrderedDict[str, list] = OrderedDict()

    def check(self):
        if not memory.cuda_ready():
            return False, "нужна видеокарта NVIDIA и PyTorch с CUDA (полный профиль установки)"
        missing = weights.missing(self.settings, "zimage")
        if missing:
            return False, weights.missing_message(missing)
        gguf = weights.folder(self.settings, "zimage_gguf") / self.settings.zimage_gguf
        if not gguf.is_file():
            return False, f"нет файла {gguf.name} в {gguf.parent}"
        return True, ""

    def load(self, ctx, force_mode: str | None = None):
        import torch
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            GGUFQuantizationConfig,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )
        from transformers import AutoTokenizer

        memory.require_ram(RAM_LOAD_GB, "Z-Image-Turbo", warn=ctx.warn)
        root = weights.folder(self.settings, "zimage")
        gguf = weights.folder(self.settings, "zimage_gguf") / self.settings.zimage_gguf
        mode = force_mode or diffusion.requested_mode(ctx) or diffusion.pick_mode(memory.vram_free_gb(), VRAM_OFFLOAD_GB, VRAM_GROUP_GB)
        ctx.progress(0.01, "загрузка Z-Image-Turbo (GGUF)")
        transformer = ZImageTransformer2DModel.from_single_file(
            str(gguf),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16,
            config=str(root),
            subfolder="transformer",
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(str(root), subfolder="vae", dtype=torch.bfloat16, local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root), subfolder="scheduler", local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), subfolder="tokenizer", local_files_only=True)
        # the main pipeline never sees the text encoder: it gets ready prompt embeddings
        self.pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
        diffusion.apply_mode(self.pipe, mode)
        if mode != "offload" or memory.vram_free_gb() < 4.0:
            self.pipe.vae.enable_tiling()
        self.mode = mode
        if self.settings.zimage_keep_text_encoder:
            self._load_encoder(ctx)
        log.info("Z-Image-Turbo loaded, memory mode %s", mode)
        if mode != "offload":
            ctx.warn(f"Мало свободной видеопамяти: Z-Image работает в режиме «{diffusion.MODE_TITLES[mode]}» (медленнее).")

    def _load_encoder(self, ctx) -> None:
        import torch
        from transformers import AutoModel

        memory.require_ram(RAM_ENCODER_GB, "текстового энкодера Z-Image", warn=ctx.warn)
        ctx.progress(0.03, "загрузка текстового энкодера (CPU)")
        root = weights.folder(self.settings, "zimage")
        self.text_encoder = AutoModel.from_pretrained(str(root), subfolder="text_encoder", dtype=torch.bfloat16, local_files_only=True)
        self.text_encoder.eval()

    def _drop_encoder(self) -> None:
        self.text_encoder = None
        gc.collect()

    def unload(self):
        self.pipe = self.tokenizer = None
        self._drop_encoder()
        self.mode = None
        memory.free_gpu()
        self.loaded = False

    def encode(self, ctx, prompt: str):
        """Prompt embeddings, computed on the CPU (exact length, no 512-token padding)."""
        import torch
        from diffusers import ZImagePipeline

        if prompt in self.embeds:
            self.embeds.move_to_end(prompt)
            return self.embeds[prompt], True
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        n = len(self.tokenizer(text).input_ids)
        if n > 512:
            raise UserError(f"Промпт слишком длинный: {n} токенов, максимум 512")
        if self.text_encoder is None:
            self._load_encoder(ctx)
        ctx.progress(0.08, "кодирование промпта (CPU)")
        encoder = ZImagePipeline(scheduler=None, vae=None, text_encoder=self.text_encoder, tokenizer=self.tokenizer, transformer=None)
        try:
            with torch.inference_mode():
                embeds, _ = encoder.encode_prompt(prompt, device=torch.device("cpu"), do_classifier_free_guidance=False, max_sequence_length=n)
        finally:
            del encoder
            if not self.settings.zimage_keep_text_encoder:
                self._drop_encoder()
        embeds = [e.detach().clone() for e in embeds]
        self.embeds[prompt] = embeds
        while len(self.embeds) > CACHE_PROMPTS:
            self.embeds.popitem(last=False)
        return embeds, False

    def run(self, ctx, params, inputs):
        import torch

        diffusion.ensure_mode(ctx, self)
        steps, width, height = params["steps"], params["width"], params["height"]
        ctx.progress(0.02, "промпт")
        embeds, cached = self.encode(ctx, params["prompt"])
        embeds = [e.to("cuda", torch.bfloat16) for e in embeds]
        if params.get("negative"):
            ctx.warn("Z-Image-Turbo не использует негативный промпт (модель без CFG).")

        def generate():
            generator = torch.Generator("cpu").manual_seed(params["seed"])
            with torch.inference_mode():
                return self.pipe(
                    prompt=None,
                    prompt_embeds=embeds,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    generator=generator,
                    callback_on_step_end=diffusion.step_callback(ctx, steps),
                ).images[0]

        image = diffusion.run_with_fallback(ctx, self, generate)
        ctx.progress(0.97, "сохранение")
        meta = {k: params[k] for k in ("prompt", "width", "height", "steps", "seed")}
        meta.update({k: params[k] for k in ("subject", "style") if params.get(k)})
        imaging.save_png(image, ctx.output("image", "png"), {**meta, "backend": self.name, "memory_mode": self.mode})
        return {"seed": params["seed"], "prompt": params["prompt"], "width": width, "height": height,
                "memory_mode": self.mode, "prompt_cached": cached, "prompt_tokens": len(embeds[0])}
