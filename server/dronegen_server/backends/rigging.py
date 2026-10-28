"""Skeletons for animation (``dronegen_server.rigging``): a job of kind ``rig``.

Input: the mesh as Blender has it now (object space, Z up), so the bones fit whatever the user did to the
model after import. Output: bones with roles and sides (JSON in the job result and in a ``rig`` file) and a
preview sheet. The ``template`` way needs no GPU and takes a second or two.
"""

from __future__ import annotations

import json

import numpy as np

from .. import rigging
from ..errors import UserError
from .base import Backend, register

MAX_VERTICES = 2_000_000


def mesh_arrays(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = inputs.get("vertices"), inputs.get("faces")
    if vertices is None or faces is None:
        raise UserError("Не передана сетка (vertices и faces)")
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise UserError("Сетка: нужны vertices (N×3) и треугольники faces (M×3)")
    if not len(vertices) or not len(faces):
        raise UserError("Пустая сетка: не из чего строить скелет")
    if len(vertices) > MAX_VERTICES:
        raise UserError(f"Слишком большая сетка: {len(vertices)} вершин (не больше {MAX_VERTICES})")
    if faces.min() < 0 or faces.max() >= len(vertices) or not np.isfinite(vertices).all():
        raise UserError("Сетка повреждена: номера вершин вне диапазона или нечисловые координаты")
    return vertices, faces


@register
class TemplateRig(Backend):
    kind = "rig"
    name = "template"
    title = "Скелет по форме модели (без нейросети)"
    defaults = {"plan": "auto"}

    def validate(self, params):
        p = super().validate(params)
        rigging.hints_of(p.get("plan") or None)  # an unknown plan fails now, not in the queue
        return p

    def run(self, ctx, params, inputs):
        vertices, faces = mesh_arrays(inputs)
        ctx.progress(0.1, "строю скелет")
        plan = params.get("plan")
        result, sheet = rigging.fit(vertices, faces, None if plan in (None, "", "auto") else plan)
        ctx.progress(0.9, "превью")
        sheet.save(ctx.output("preview", "png"))
        ctx.output("rig", "json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return {**result, "method": self.name}
