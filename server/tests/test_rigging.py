"""Skeletons for animation (``dronegen_server.rigging``) on figures made of primitives, in Blender axes (Z up,
facing -Y like an imported DroneGen model): which part is what, left and right, bones inside the body."""

import numpy as np
import pytest
import trimesh

from dronegen_server import rigging
from dronegen_server.parametric import shapes as S
from dronegen_server.rigging import roles, skeleton


def solid(*parts) -> trimesh.Trimesh:
    return trimesh.util.concatenate(list(parts))


def capsule(a, b, r, sections=16):
    return solid(S.cylinder(a, b, r, sections), S.sphere(a, r, sections), S.sphere(b, r, sections))


def plate(center, size, tilt_deg=0.0, axis=(0, 1, 0)):
    return S.rotate(S.box(center, size), tilt_deg, axis, center)


def quadruped(lift_front_right=False, wings=False):
    """A horse-like figure facing -Y: body, four legs to the ground, a neck with a head up front, a thin tail."""
    parts = [capsule((0, 0.6, 1.0), (0, -0.6, 1.0), 0.3)]
    for x in (-0.2, 0.2):
        for y in (-0.45, 0.45):
            foot = (x, y - 0.35, 0.45) if (lift_front_right and x < 0 and y < 0) else (x, y, 0.06)
            parts.append(capsule((x, y, 0.9), foot, 0.08))
    parts += [capsule((0, -0.55, 1.1), (0, -0.85, 1.65), 0.15), capsule((0, -0.85, 1.65), (0, -1.25, 1.55), 0.13),
              capsule((0, 0.75, 1.0), (0, 1.25, 0.55), 0.045)]
    if wings:  # thin feathered plates rising from the shoulders
        parts += [plate((0.8, -0.35, 1.45), (1.3, 0.5, 0.06), -25), plate((-0.8, -0.35, 1.45), (1.3, 0.5, 0.06), 25)]
    return solid(*parts)


def bird():
    """A perched bird seen from the front: upright body, a head, wings spread to the sides, legs with toes forward."""
    parts = [capsule((0, 0, 0.55), (0, 0, 1.15), 0.24), capsule((0, 0, 1.15), (0, -0.05, 1.45), 0.13),
             plate((0.85, 0, 1.25), (1.2, 0.45, 0.05), -20), plate((-0.85, 0, 1.25), (1.2, 0.45, 0.05), 20),
             plate((0, 0.38, 0.3), (0.3, 0.55, 0.05), -40, (1, 0, 0))]
    for x in (-0.11, 0.11):
        parts += [capsule((x, 0, 0.6), (x, 0, 0.06), 0.05), capsule((x, 0, 0.05), (x, -0.22, 0.05), 0.04)]
    return solid(*parts)


def swordsman():
    """A person in a T-pose holding a sword out in the right hand (the right is -X for a figure facing -Y)."""
    parts = [capsule((0, 0, 0.95), (0, 0, 1.4), 0.17), capsule((0, 0, 1.45), (0, 0, 1.72), 0.11)]
    for x in (-0.1, 0.1):
        parts += [capsule((x, 0, 0.95), (x, 0, 0.08), 0.07), capsule((x, 0, 0.06), (x, -0.18, 0.06), 0.05)]
    parts += [capsule((0.1, 0, 1.38), (0.75, 0, 1.38), 0.05), capsule((-0.1, 0, 1.38), (-0.75, 0, 1.38), 0.05),
              S.box((-1.15, 0, 1.38), (0.8, 0.03, 0.07))]
    return solid(*parts)


def fit(mesh, plan=None):
    result, sheet = rigging.fit(mesh.vertices, mesh.faces, plan)
    return result, {b["name"]: b for b in result["bones"]}, sheet


def chain_tip(bones, chain):
    last = max((b for b in bones.values() if b["chain"] == chain), key=lambda b: b["segment"])
    return np.array(last["tail"])


def test_quadruped_has_four_legs_head_in_front_and_a_tail():
    result, bones, sheet = fit(quadruped())
    assert result["parts"].get("leg_front") == 2 and result["parts"].get("leg_hind") == 2
    assert not result["upright"]
    assert np.allclose(result["frame"]["forward"], [0, -1, 0], atol=0.15)
    assert np.allclose(result["frame"]["left"], [1, 0, 0], atol=0.15)  # facing -Y, the figure's left is +X
    assert chain_tip(bones, "leg_front.L")[0] > 0 and chain_tip(bones, "leg_front.R")[0] < 0
    assert chain_tip(bones, "leg_front.L")[1] < 0 < chain_tip(bones, "leg_hind.L")[1]
    for name in ("leg_front.L", "leg_hind.R"):
        assert chain_tip(bones, name)[2] < 0.25  # down to the ground
        assert len([b for b in bones.values() if b["chain"] == name]) == 4  # thigh, shin, cannon, hoof
    head = np.array(bones["head"]["tail"])
    assert head[1] < -1.0 and head[2] > 1.3  # the muzzle, in front and up
    assert np.array(bones["tail_1"]["head"])[1] > 0.3  # the tail grows at the back
    assert bones["root"]["parent"] is None and not bones["root"]["deform"]
    assert abs(bones["root"]["head"][2] - result["ground"]) < 1e-6
    assert sheet.width > sheet.height > 50


def test_bones_form_one_tree_of_connected_chains():
    _, bones, _ = fit(quadruped(wings=True))
    for b in bones.values():
        if b["parent"] is not None:
            assert b["parent"] in bones
            parent = bones[b["parent"]]
            if parent["chain"] == b["chain"]:  # inside a chain each bone starts where the previous ends
                assert parent["segment"] == b["segment"] - 1
                assert np.allclose(parent["tail"], b["head"], atol=1e-6)
    roots = [b for b in bones.values() if b["parent"] is None]
    assert [b["name"] for b in roots] == ["root"]
    assert {bones[n]["parent"] for n in ("leg_hind_1.L", "leg_hind_1.R")} == {"spine_1"}  # hind legs on the hips
    assert bones["wing_1.L"]["parent"] == bones["leg_front_1.L"]["parent"] == "spine_3"  # wings and arms on the chest


def test_bones_lie_inside_the_body():
    mesh = quadruped(wings=True)
    _, bones, _ = fit(mesh)
    vol = skeleton.volume(mesh.vertices, mesh.faces)
    for b in bones.values():
        if not b["deform"]:
            continue
        for p in (b["head"], b["tail"], (np.array(b["head"]) + b["tail"]) / 2):
            ijk = np.round((np.asarray(p) - vol.origin) / vol.pitch).astype(int)
            near = vol.grid[tuple(slice(max(0, i - 1), i + 2) for i in ijk)]
            assert near.any(), f"{b['name']} leaves the body at {p}"


def test_flat_plates_are_wings_left_and_right():
    result, bones, _ = fit(quadruped(wings=True))
    assert result["parts"].get("wing") == 2
    assert chain_tip(bones, "wing.L")[0] > 1.0 and chain_tip(bones, "wing.R")[0] < -1.0
    assert result["parts"].get("leg_front") == 2 and result["parts"].get("leg_hind") == 2


def test_a_lifted_leg_needs_the_body_plan():
    auto, _, _ = fit(quadruped(lift_front_right=True))
    planned, bones, _ = fit(quadruped(lift_front_right=True), "quadruped")
    assert sum(v for k, v in auto["parts"].items() if k.startswith("leg")) == 3
    assert planned["parts"].get("leg_front") == 2 and planned["parts"].get("leg_hind") == 2
    assert chain_tip(bones, "leg_front.R")[2] > 0.3  # the lifted one


def test_turned_figure_faces_where_its_head_is():
    turned = quadruped().copy()
    turned.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1)))  # now facing +X
    result, bones, _ = fit(turned)
    assert np.allclose(result["frame"]["forward"], [1, 0, 0], atol=0.15)
    assert chain_tip(bones, "leg_front.L")[1] > 0  # its left is +Y now


def test_standing_bird_with_wings_and_legs():
    result, bones, _ = fit(bird())
    assert result["upright"]
    assert result["parts"].get("wing") == 2 and result["parts"].get("leg") == 2
    assert np.allclose(result["frame"]["forward"], [0, -1, 0], atol=0.2)  # the toes point forward
    assert np.array(bones["head"]["tail"])[2] > 1.35
    assert chain_tip(bones, "wing.L")[0] > 1.0
    tail = chain_tip(bones, "tail")  # the fan tail behind and below, hung from the hips
    assert tail[1] > 0.2 and tail[2] < 0.5 and bones["tail_1"]["parent"] == "spine_1"


def test_swordsman_has_arms_and_a_sword_in_the_right_hand():
    result, bones, _ = fit(swordsman(), "biped")
    assert result["upright"] and result["parts"].get("arm") == 2 and result["parts"].get("leg") == 2
    assert "prop.R" in bones and bones["prop.R"]["parent"] == "arm_3.R"
    assert bones["prop.R"]["tail"][0] < -1.3  # the blade's tip
    assert chain_tip(bones, "arm.L")[0] > 0.6


def test_rigid_object_is_one_chain():
    dombra = capsule((0, 0, 0.2), (0, 0, 1.6), 0.08)
    result, bones, _ = fit(solid(dombra, S.sphere((0, 0, 0.35), 0.3)), "rigid")
    assert result["rigid"] and result["parts"] == {}
    assert set(bones) == {"root"} | {b for b in bones if b.startswith("body")}
    assert all(bones[b]["parent"] in bones for b in bones if b != "root")


def test_unknown_body_plan_is_refused():
    from dronegen_server.errors import UserError

    with pytest.raises(UserError, match="план тела"):
        rigging.fit(quadruped().vertices, quadruped().faces, "octopus")
    assert "winged_quadruped" in roles.BODY_PLANS


def npz_bytes(mesh) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, vertices=np.asarray(mesh.vertices, np.float32), faces=np.asarray(mesh.faces, np.int32))
    return buf.getvalue()


def test_rig_api_round_trip(client):
    from conftest import wait_job

    assert any(b["name"] == "template" and b["available"] for b in client.get("/health").json()["backends"]["rig"])
    plans = client.get("/rig/plans").json()
    assert plans["winged_quadruped"]["hints"]["wings"] == 2 and "тулпар" in plans["winged_quadruped"]["title"]
    r = client.post("/rig", files={"mesh": ("horse.npz", npz_bytes(quadruped(wings=True)))},
                    data={"plan": "winged_quadruped"})
    assert r.status_code == 200, r.text
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    result = job["result"]
    assert result["method"] == "template" and result["parts"]["wing"] == 2
    assert {"name", "role", "side", "chain", "segment", "head", "tail", "parent", "deform"} <= set(result["bones"][0])
    rig = client.get(f"/files/{job['files']['rig']}").json()
    assert [b["name"] for b in rig["bones"]] == [b["name"] for b in result["bones"]]
    assert client.get(f"/files/{job['files']['preview']}").headers["content-type"] == "image/png"


def test_rig_api_refuses_bad_input(client):
    assert client.post("/rig", data={"plan": "auto"}).status_code == 400
    assert client.post("/rig", files={"mesh": ("x.npz", b"not a zip")}).status_code == 400
    r = client.post("/rig", files={"mesh": ("x.npz", npz_bytes(quadruped()))}, data={"plan": "octopus"})
    assert r.status_code == 400 and "план тела" in r.json()["detail"]
    import io

    buf = io.BytesIO()
    np.savez(buf, vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 7]]))
    r = client.post("/rig", files={"mesh": ("x.npz", buf.getvalue())})
    assert r.status_code == 200  # checked in the job: a broken mesh is a readable job error
    from conftest import wait_job

    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "error" and "повреждена" in job["error"]
