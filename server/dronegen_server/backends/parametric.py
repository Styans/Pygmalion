"""Exact models from a drawing (``dronegen_server.parametric``): no picture, no neural network, no GPU.

A job of kind ``build``: ``{"template": "dombra", "values": {"length": 1.1}}`` -> a GLB with
vertex colours per part, an OBJ and a preview sheet. A build takes tens of milliseconds.
"""

from __future__ import annotations

from .. import meshing, parametric, preview, quality
from ..errors import UserError
from .base import Backend, register


@register
class Parametric(Backend):
    kind = "build"
    name = "parametric"
    title = "Точный чертёж (без нейросети)"
    defaults = {"template": "dombra", "values": {}}

    def validate(self, params):
        p = super().validate(params)
        if not isinstance(p.get("values") or {}, dict):
            raise UserError("values: нужен объект {параметр: число}")
        p["values"] = dict(p.get("values") or {})
        parametric.resolve(p["template"], p["values"])  # unknown names and out-of-range numbers fail now
        return p

    def run(self, ctx, params, inputs):
        ctx.progress(0.2, "строю по чертежу")
        mesh, values = parametric.build(params["template"], params["values"])
        meshing.normalize(mesh)
        ctx.progress(0.6, "превью и экспорт")
        preview.sheet(mesh).save(ctx.output("preview", "png"))
        meshing.export(mesh, ctx.output("mesh", "glb"), ctx.output("mesh_obj", "obj"))
        template = parametric.get(params["template"])
        return {**meshing.stats(mesh), "template": params["template"], "title": template.title, "values": values,
                "quality": quality.report(mesh)}
