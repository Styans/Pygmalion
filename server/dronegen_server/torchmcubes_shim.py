"""Stand-in for ``torchmcubes``, which TripoSR imports in ``tsr/models/isosurface.py``.

The real package is built from git with a CUDA toolchain and often fails on Windows. This
module offers the same ``marching_cubes(volume, threshold) -> (verts, faces)`` on the CPU
(PyMCubes, or scikit-image), with torchmcubes' output convention:

* vertices are (x, y, z) = (last axis, middle axis, first axis) indices, which is why TripoSR
  reorders them with ``v[..., [2, 1, 0]]``;
* ``volume > threshold`` is inside, and after TripoSR's reordering the triangles face outwards.
"""

from __future__ import annotations

import numpy as np
import torch

from . import meshing


def marching_cubes(vol: torch.Tensor, thresh: float):
    volume = vol.detach().float().cpu().numpy()
    verts, faces = meshing.marching_cubes(volume, float(thresh))  # (i, j, k), wound outwards
    # TripoSR turns (k, j, i) back into (i, j, k): the faces stay valid as they are
    verts = np.ascontiguousarray(verts[:, ::-1], dtype=np.float32)
    return torch.from_numpy(verts), torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64))


def grid_interp(*_args, **_kwargs):
    raise NotImplementedError("grid_interp из torchmcubes TripoSR не использует")
