"""Skeleton fitting on real generated models: does it find the parts a person sees?

    runtime\\venv\\Scripts\\python tests\\rig_benchmark.py          # the server code of this folder

Each case is a model from runtime/outputs (generated on this computer; missing ones are skipped) with what
it should get: legs, wings, arms, standing or lying, where it faces. Tuning one rule of
``dronegen_server.rigging.roles`` on one model easily breaks another: run this after every change
(it was 50 of 50 when the rules were written, 2026-10-21).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

OUTPUTS = ROOT / "runtime" / "outputs"
# job, body plan, legs, wings (None: any), arms, standing, facing (x, y)
CASES = [
    ("a1a8689d795c", None, 4, 2, 0, False, (0, -1)),  # white pegasus, front view
    ("3e873c90b712", None, 4, 0, 0, False, (0, -1)),  # brown horse, front view
    ("2c17d0e3a22c", None, 4, 0, 0, False, (0, -1)),  # horse, front view
    ("9cd97aaa757e", None, 4, None, 0, False, (-1, 0)),  # golden pegasus galloping, side view (a wing hidden)
    ("7525008abfed", "quadruped", 4, 0, 0, False, (-1, 0)),  # snow leopard leaping, one paw in the air
    ("5cd35e7fbdca", None, 2, 2, 0, True, (0, -1)),  # eagle, wings spread
    ("cca51d7e8c02", None, 2, 2, 0, True, (0, -1)),  # white eagle, wings spread
    ("c6277c7cbe39", None, 2, 2, 0, True, (0, -1)),  # penguin-like bird, its tail on the ground
    ("5cd35e7fbdca", "bird", 2, 2, 0, True, (0, -1)),
    ("e4e39bf6ba37", "biped", 2, 0, 2, True, (0, -1)),  # cartoon turtle with a shell
]


def load(job: str):
    import trimesh

    mesh = trimesh.load(OUTPUTS / f"{job}_mesh.glb", force="mesh", process=False)
    v = np.asarray(mesh.vertices)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1), np.asarray(mesh.faces)  # glTF -> Blender axes


def main() -> int:
    from dronegen_server import rigging

    for stream in (sys.stdout, sys.stderr):  # Russian survives pipes on Windows
        stream.reconfigure(encoding="utf-8", errors="replace")
    total = good = 0
    for job, plan, legs, wings, arms, upright, facing in CASES:
        if not (OUTPUTS / f"{job}_mesh.glb").is_file():
            print(f"{job}: нет модели, пропущено")
            continue
        t0 = time.time()
        result, _ = rigging.fit(*load(job), plan)
        parts = result["parts"]
        got_legs = sum(n for k, n in parts.items() if k.startswith("leg"))
        checks = {
            "ноги": got_legs == legs,
            "крылья": wings is None or parts.get("wing", 0) == wings,
            "руки": parts.get("arm", 0) == arms,
            "стоит": result["upright"] == upright,
            "смотрит": float(np.dot(result["frame"]["forward"][:2], facing)) > 0.8,
        }
        total += len(checks)
        good += sum(checks.values())
        bad = [k for k, ok in checks.items() if not ok]
        print(f"{job} {plan or 'auto':10s} {time.time() - t0:4.1f} с  ноги {got_legs} крылья {parts.get('wing', 0)} "
              f"руки {parts.get('arm', 0)} стоит {result['upright']!s:5}  {'OK' if not bad else 'НЕ ТАК: ' + ', '.join(bad)}")
    print(f"ИТОГ: {good} из {total}")
    return 0 if good == total else 1


if __name__ == "__main__":
    sys.exit(main())
