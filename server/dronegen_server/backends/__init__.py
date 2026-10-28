"""Backends register themselves on import (see ``base.register``).

All modules here import only light libraries at the top: torch and diffusers are imported
inside ``load()`` / ``run()``, so the server starts (and the CPU backends work) without them.
"""

from . import hunyuan, mock, parametric, rigging, sdxl, silhouette, triposr, zimage  # noqa: F401
from .base import Backend, Img2MeshBackend, Registry, Txt2ImgBackend, register  # noqa: F401
