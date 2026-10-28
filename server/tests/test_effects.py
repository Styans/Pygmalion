"""Effects with drones of their own (``dronegen_server.effects``): shapes keep the safety distance, every drone has
a waiting place in the reserve, and flights from the reserve are short and do not cross."""

import numpy as np
import pytest

from dronegen_server import effects
from dronegen_server.errors import UserError

FRAME = {"forward": [0, -1, 0], "left": [1, 0, 0], "up": [0, 0, 1]}  # a figure facing the audience at -Y
BEHIND = [0, 30, 0]


def closest(points) -> float:
    p = np.asarray(points)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def crossings(a, b) -> int:
    """Pairs of straight flights a[i] -> b[i] that cross, seen from the audience (the x-z plane)."""
    a, b = np.asarray(a)[:, [0, 2]], np.asarray(b)[:, [0, 2]]

    def side(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    n, count = len(a), 0
    for i in range(n):
        j = np.arange(i + 1, n)
        cross = (side(a[i], b[i], a[j]) * side(a[i], b[i], b[j]) < 0) & (side(a[j], b[j], a[i]) * side(a[j], b[j], b[i]) < 0)
        count += int(cross.sum())
    return count


@pytest.mark.parametrize("kind", ["star", "ring", "heart", "ripple"])
def test_outlines_fit_as_many_drones_as_the_safety_distance_allows(kind):
    out = effects.build(kind, size=20.0, spacing=2.0, frame=FRAME, reserve_at=BEHIND)
    shape, waiting = np.asarray(out["shape"]), np.asarray(out["reserve"])
    assert out["count"] == len(shape) == len(waiting) > 5
    assert closest(shape) >= 2.0 - 1e-6 and closest(waiting) >= 2.0 - 1e-6
    flat = 2 if kind == "ripple" else 1  # a ripple on the level, the rest in the plane the audience sees
    assert np.allclose(shape[:, flat], 0)
    assert np.ptp(shape[:, 0]) <= 20.0 + 1e-6 or kind == "ripple"
    assert np.allclose(waiting.mean(axis=0), BEHIND, atol=2.0)  # the reserve waits where it was asked


def test_ripple_starts_small_and_says_how_far_it_spreads():
    out = effects.build("ripple", size=30.0, spacing=3.0, frame=FRAME, reserve_at=BEHIND)
    shape = np.asarray(out["shape"])
    assert np.allclose(shape[:, 2], 0)  # on the level
    start_r = np.linalg.norm(shape[:, :2], axis=1).mean()
    assert start_r < 0.5 * 15.0 and out["params"]["expand"] > 0.5
    assert start_r * (1 + out["params"]["expand"]) == pytest.approx(15.0, rel=0.02)


def test_too_many_drones_for_the_size_are_cut_and_reported():
    out = effects.build("star", size=10.0, spacing=2.0, frame=FRAME, reserve_at=BEHIND, count=500)
    assert out["count"] < 500 and "помещается" in out["notes"][0]
    with pytest.raises(UserError, match="мал"):
        effects.build("ring", size=1.0, spacing=3.0, frame=FRAME, reserve_at=BEHIND)
    with pytest.raises(UserError, match="мал"):  # four drones are no star: its tips would take the rest
        effects.build("star", size=4.0, spacing=3.0, frame=FRAME, reserve_at=BEHIND)
    assert effects.build("star", size=12.0, spacing=3.0, frame=FRAME, reserve_at=BEHIND)["count"] >= effects.MIN_OUTLINE


def test_sparkles_and_splash_keep_apart_and_splash_flies_out_and_up():
    sparks = effects.build("sparkles", size=20.0, spacing=3.0, frame=FRAME, reserve_at=BEHIND, count=40)
    assert sparks["count"] == 40 and closest(sparks["shape"]) >= 3.0 - 1e-6
    assert np.linalg.norm(np.asarray(sparks["shape"]), axis=1).max() <= 10.0 + 1e-6
    splash = effects.build("splash", size=10.0, spacing=2.0, frame=FRAME, reserve_at=BEHIND, count=20,
                           params={"speed": 4.0})
    v = np.asarray(splash["velocity"])
    assert len(v) == splash["count"] == 20 and closest(splash["shape"]) >= 2.0 - 1e-6
    assert np.allclose(np.linalg.norm(v, axis=1), 4.0) and (v[:, 2] > 0).all()  # out and up at the speed asked


def test_reserve_flights_are_short_and_do_not_cross():
    out = effects.build("star", size=30.0, spacing=2.5, frame=FRAME, reserve_at=[0, 40, 0])
    a, b = np.asarray(out["reserve"]), np.asarray(out["shape"])
    assert crossings(a, b) == 0
    shuffled = a[np.random.default_rng(1).permutation(len(a))]
    assert np.linalg.norm(a - b, axis=1).sum() < np.linalg.norm(shuffled - b, axis=1).sum()


def test_given_points_are_thinned_to_the_safety_distance():
    path = np.stack([np.linspace(0, 20, 200), np.zeros(200), np.sin(np.linspace(0, 3, 200)) * 5], 1)
    out = effects.build("points", size=20.0, spacing=2.0, frame=FRAME, reserve_at=BEHIND, points=path.tolist())
    assert out["count"] < 200 and closest(out["shape"]) >= 2.0 - 1e-6 and "оставлено" in out["notes"][0]
    with pytest.raises(UserError, match="пусто"):
        effects.build("points", size=20.0, spacing=2.0, frame=FRAME, reserve_at=BEHIND, points=[])


def test_effects_api(client):
    assert client.get("/effects").json()["star"] == "Звезда"
    r = client.post("/effects/build", json={"kind": "heart", "size": 12, "spacing": 1.5, "frame": FRAME,
                                            "reserve_at": BEHIND})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == len(r.json()["shape"]) > 10
    r = client.post("/effects/build", json={"kind": "dragon", "size": 12, "spacing": 1.5, "frame": FRAME,
                                            "reserve_at": BEHIND})
    assert r.status_code == 400 and "Нет эффекта" in r.json()["detail"]
