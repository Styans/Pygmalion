"""Motions for skeletons (``rigging.motions``): every motion bakes on a skeleton that has its parts, cycles close,
and the parts go where the motion says (checked by forward kinematics of the baked keys)."""

import numpy as np
import pytest
from test_rigging import bird, quadruped, swordsman

from dronegen_server import rigging
from dronegen_server.errors import UserError
from dronegen_server.rigging import motions

_cache = {}


def skeleton_of(name: str, plan=None) -> tuple[motions.Skeleton, dict]:
    """A synthetic figure's skeleton with rest matrices, as the Blender add-on would send it."""
    if name not in _cache:
        mesh = {"horse": lambda: quadruped(wings=True), "plain_horse": quadruped, "bird": bird,
                "swordsman": swordsman}[name]()
        result, _ = rigging.fit(mesh.vertices, mesh.faces, plan)
        data = {"frame": result["frame"], "size": result["size"],
                "bones": motions.rest_matrices(result["bones"], result["frame"])}
        _cache[name] = (motions.skeleton(data), data)
    return _cache[name]


def tip(sk, baked, chain, index):
    last = max((b for b in sk.bones if b.chain == chain), key=lambda b: b.segment)
    return motions.posed_tails(sk, baked, index)[last.name]


def rest_tip(sk, chain):
    return max((b for b in sk.bones if b.chain == chain), key=lambda b: b.segment).tail


def at(baked, t):
    return int(np.argmin(np.abs(np.asarray(baked["times"]) - t)))


def test_catalog_is_complete_and_readable():
    cat = motions.catalog()
    assert {"wing_flap", "gallop", "walk", "trot", "walk_biped", "tail_wave", "breathe", "look_around", "spin",
            "wings_up", "rear_up", "head_left", "sword_swing", "jump"} <= set(cat)
    for key, m in cat.items():
        assert m["title"] and m["about"] and m["roles"] and m["kind"] in ("cycle", "pose", "once"), key
        assert (m["period"] > 0) == (m["kind"] != "pose"), key
        for name, p in m["params"].items():
            assert p["min"] <= p["default"] <= p["max"] and p["title"], (key, name)


@pytest.mark.parametrize("key", sorted(motions.MOTIONS))
def test_every_motion_bakes_on_a_figure_that_has_its_parts(key):
    m = motions.MOTIONS[key]
    figure = "swordsman" if {"arm", "leg"} & set(m.roles) and not {"leg_front"} & set(m.roles) else "horse"
    sk, _ = skeleton_of(figure)
    baked = motions.bake(sk, key)
    assert baked["kind"] == m.kind and baked["bones"] or baked["root"]
    n = len(baked["times"])
    assert n == (1 if m.kind == "pose" else baked["span"] * motions.SAMPLES + 1)
    for name, keys in baked["bones"].items():
        q = np.asarray(keys)
        assert q.shape == (n, 4) and np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5), name
        if m.kind == "cycle":
            assert np.allclose(q[0], q[-1]), f"{key}: {name} does not close its loop"
        assert all(float(a @ b) >= 0 for a, b in zip(q[:-1], q[1:])), f"{key}: {name} flips between keys"
        # on the near side: Blender mixes a part of the turn (fades), and -q would go almost 360° round
        assert q[0][0] >= 0, f"{key}: {name} starts with w < 0"
        if key != "spin":
            assert q[:, 0].min() > 0.3, f"{key}: {name} turns more than ~145°"


def test_wing_flap_lifts_and_lowers_both_wings_alike():
    sk, _ = skeleton_of("horse")
    baked = motions.bake(sk, "wing_flap", {"amplitude": 40})
    for chain in ("wing.L", "wing.R"):
        rest = rest_tip(sk, chain)[2]
        assert tip(sk, baked, chain, at(baked, 0.25))[2] > rest + 0.2  # up a quarter in
        assert tip(sk, baked, chain, at(baked, 0.75))[2] < rest - 0.2  # down three quarters in
        assert np.allclose(tip(sk, baked, chain, 0), tip(sk, baked, chain, len(baked["times"]) - 1), atol=1e-6)
    up_l, up_r = tip(sk, baked, "wing.L", at(baked, 0.25)), tip(sk, baked, "wing.R", at(baked, 0.25))
    rise_l, rise_r = up_l[2] - rest_tip(sk, "wing.L")[2], up_r[2] - rest_tip(sk, "wing.R")[2]
    assert abs(rise_l - rise_r) < 0.1 * rise_l and up_l[0] > 0 > up_r[0]  # a mirror pair: rise alike, on both sides
    assert np.allclose(baked["root"]["location"][0], baked["root"]["location"][-1])


def test_walk_moves_each_leg_forward_in_its_turn():
    sk, _ = skeleton_of("plain_horse")
    baked = motions.bake(sk, "walk", {"stride": 30})
    forward = sk.forward
    for chain, phase in (("leg_hind.L", 0.0), ("leg_front.L", 0.25), ("leg_hind.R", 0.5), ("leg_front.R", 0.75)):
        rest = rest_tip(sk, chain) @ forward
        ahead = tip(sk, baked, chain, at(baked, (phase + 0.25) % 1.0)) @ forward
        behind = tip(sk, baked, chain, at(baked, (phase + 0.75) % 1.0)) @ forward
        assert ahead > rest + 0.1 and behind < rest - 0.1, chain


def test_head_turns_left_and_the_figure_rears_up():
    sk, _ = skeleton_of("plain_horse")
    head = next(b for b in sk.bones if b.role == "head")
    left = motions.posed_tails(sk, motions.bake(sk, "head_left"), 0)[head.name]
    assert (left - head.tail) @ sk.left > 0.1
    reared = motions.bake(sk, "rear_up")
    tails = motions.posed_tails(sk, reared, 0)
    assert tails[head.name][2] > head.tail[2] + 0.3  # the head goes up
    assert tip(sk, reared, "leg_front.L", 0)[2] > rest_tip(sk, "leg_front.L")[2] + 0.2  # front feet off the ground
    assert abs(tip(sk, reared, "leg_hind.L", 0)[2] - rest_tip(sk, "leg_hind.L")[2]) < 0.15  # hind feet stay down


def test_spin_turns_the_whole_figure_about_the_vertical():
    sk, _ = skeleton_of("plain_horse")
    baked = motions.bake(sk, "spin")
    assert baked["span"] == 2  # a whole turn is -q: the keys go round twice to close the loop
    head = next(b for b in sk.bones if b.role == "head")
    root = sk.root.head
    quarter = motions.posed_tails(sk, baked, at(baked, 0.25))[head.name]
    a, b = (head.tail - root)[:2], (quarter - root)[:2]
    angle = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
    assert angle == pytest.approx(90, abs=3)
    assert np.allclose(motions.posed_tails(sk, baked, len(baked["times"]) - 1)[head.name], head.tail, atol=1e-4)


def test_sword_swing_raises_then_strikes_with_the_sword_arm():
    sk, _ = skeleton_of("swordsman", "biped")
    baked = motions.bake(sk, "sword_swing")
    assert set(baked["bones"]) & {b.name for b in sk.bones if b.side == "R"}
    assert not set(baked["bones"]) & {b.name for b in sk.bones if b.side == "L"}
    blade = motions.posed_tails(sk, baked, at(baked, 0.35))["prop.R"]
    struck = motions.posed_tails(sk, baked, at(baked, 0.55))["prop.R"]
    rest = sk.by_name["prop.R"].tail
    assert blade[2] > rest[2] + 0.4 and struck[2] < rest[2] - 0.2


def test_jump_lifts_the_figure_by_its_height():
    sk, _ = skeleton_of("plain_horse")
    baked = motions.bake(sk, "jump", {"height": 0.5})
    shift = np.asarray(baked["root"]["location"])
    lift = shift @ sk.root.rest.T @ sk.up  # back to armature space, along up
    assert lift.max() == pytest.approx(0.5 * sk.size, rel=0.02) and abs(lift[0]) < 1e-9 and abs(lift[-1]) < 1e-9


def test_motion_needs_its_parts_and_valid_parameters():
    sk, _ = skeleton_of("plain_horse")
    with pytest.raises(UserError, match="нет крыльев"):
        motions.bake(sk, "wing_flap")
    with pytest.raises(UserError, match="от 2 до 80"):
        motions.bake(skeleton_of("horse")[0], "wing_flap", {"amplitude": 200})
    with pytest.raises(UserError, match="нет параметра"):
        motions.bake(skeleton_of("horse")[0], "wing_flap", {"speed": 1})
    with pytest.raises(UserError, match="Нет движения"):
        motions.bake(sk, "moonwalk")


def test_motion_api(client):
    cat = client.get("/motions").json()
    assert cat["gallop"]["title"] == "Галоп" and "leg_front" in cat["gallop"]["roles"]
    _, data = skeleton_of("horse")
    r = client.post("/motion/bake", json={"skeleton": data, "motion": "wing_flap", "params": {"amplitude": 30},
                                           "samples": 12})
    assert r.status_code == 200, r.text
    baked = r.json()
    assert len(baked["times"]) == 13 and baked["params"]["amplitude"] == 30
    assert any(name.startswith("wing") for name in baked["bones"])
    r = client.post("/motion/bake", json={"skeleton": data, "motion": "walk_biped"})
    assert r.status_code == 400 and "нет ног" in r.json()["detail"]
