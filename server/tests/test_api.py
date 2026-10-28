import io
import json
import time

import numpy as np
import pytest
import trimesh
from conftest import png_bytes, ring_image, wait_job
from PIL import Image


def test_health(client):
    h = client.get("/health").json()
    assert h["name"] == "dronegen"
    names = {k: {b["name"]: b for b in v} for k, v in h["backends"].items()}
    assert names["txt2img"]["mock"]["available"]
    assert names["img2mesh"]["silhouette"]["available"]
    assert "available" in h["gpu"] and "total_gb" in h["memory"]
    assert h["queue"] == {"pending": 0, "running": None}


def test_txt2img_mock_round_trip(client):
    r = client.post("/txt2img", json={"prompt": "тулпар", "width": 320, "height": 256, "steps": 3, "seed": 7, "backend": "mock"})
    assert r.status_code == 200, r.text
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    assert job["result"]["seed"] == 7
    assert job["params"]["width"] == 320 and job["params"]["height"] == 256
    f = client.get(f"/files/{job['files']['image']}")
    assert f.status_code == 200
    img = Image.open(io.BytesIO(f.content))
    assert img.size == (320, 256)
    assert "тулпар" in img.info["dronegen"]
    assert job["metrics"]["seconds"] > 0


def test_same_seed_same_picture_random_seed_reported(client):
    def make(seed):
        job = wait_job(client, client.post("/txt2img", json={"prompt": "x", "seed": seed, "steps": 1, "backend": "mock", "width": 256, "height": 256}).json()["job_id"])
        return job, client.get(f"/files/{job['files']['image']}").content

    (a, pa), (b, pb) = make(5), make(5)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(pa))), np.asarray(Image.open(io.BytesIO(pb))))
    c, _ = make(-1)
    assert c["result"]["seed"] >= 0


@pytest.mark.parametrize(
    "body, text",
    [
        ({"prompt": "x", "backend": "nope"}, "Нет бэкенда"),
        ({"prompt": "   ", "backend": "mock"}, "Пустой промпт"),
        ({"prompt": "x", "backend": "mock", "width": 100}, "Неверные параметры"),
        ({"prompt": "x", "backend": "mock", "steps": 0}, "Неверные параметры"),
    ],
)
def test_txt2img_bad_requests(client, body, text):
    r = client.post("/txt2img", json=body)
    assert r.status_code == 400
    assert text in r.json()["detail"]


def test_size_rounded_to_64(client):
    job = wait_job(client, client.post("/txt2img", json={"prompt": "x", "backend": "mock", "width": 700, "height": 300, "steps": 1}).json()["job_id"])
    assert (job["params"]["width"], job["params"]["height"]) == (704, 320)


def test_img2mesh_silhouette_upload(client):
    files = {"image": ("ring.png", png_bytes(ring_image()), "image/png")}
    r = client.post("/img2mesh", files=files, data={"backend": "silhouette", "mc_resolution": "96", "remove_bg": "false"})
    assert r.status_code == 200, r.text
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["watertight"] and job["result"]["background"] == "simple"
    glb = client.get(f"/files/{job['files']['mesh']}")
    mesh = trimesh.load(io.BytesIO(glb.content), file_type="glb", force="mesh")
    assert len(mesh.faces) > 1000
    assert mesh.visual.kind == "vertex"
    assert client.get(f"/files/{job['files']['mesh_obj']}").status_code == 200
    assert client.get(f"/files/{job['files']['cutout']}").status_code == 200


def test_img2mesh_reuses_server_image(client):
    img_job = wait_job(client, client.post("/txt2img", json={"prompt": "x", "backend": "mock", "seed": 1, "steps": 1, "width": 256, "height": 256}).json()["job_id"])
    r = client.post("/img2mesh", data={"backend": "mock", "image_name": img_job["files"]["image"]})
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["params"]["source"] == img_job["files"]["image"]


def test_img2mesh_rejects_non_images(client):
    r = client.post("/img2mesh", files={"image": ("x.png", b"not a picture", "image/png")}, data={"backend": "silhouette"})
    assert r.status_code == 400
    assert "картинку" in r.json()["detail"]
    r = client.post("/img2mesh", data={"backend": "silhouette"})
    assert r.status_code == 400


def test_empty_picture_is_a_readable_job_error(client):
    blank = Image.new("RGB", (200, 200), "white")
    r = client.post("/img2mesh", files={"image": ("w.png", png_bytes(blank), "image/png")}, data={"backend": "silhouette", "remove_bg": "false"})
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "error"
    assert "не найден объект" in job["error"]


@pytest.mark.parametrize("name", ["..%2Fconfig.json", "%2e%2e", "nope.png", ".hidden"])
def test_files_are_confined_to_outputs(client, settings, name):
    (settings.home / "config.json").write_text("{}")
    assert client.get(f"/files/{name}").status_code == 404


def test_unknown_job(client):
    assert client.get("/jobs/doesnotexist").status_code == 404
    assert client.post("/jobs/doesnotexist/cancel").status_code == 404


def test_queue_runs_one_job_at_a_time_and_cancels(client):
    slow = client.post("/txt2img", json={"prompt": "x", "backend": "mock", "steps": 60, "width": 256, "height": 256}).json()
    queued = client.post("/txt2img", json={"prompt": "y", "backend": "mock", "steps": 1, "width": 256, "height": 256}).json()
    assert queued["queue_position"] >= 1
    assert client.post(f"/jobs/{queued['job_id']}/cancel").json()["status"] == "cancelled"
    time.sleep(0.3)
    running = client.get(f"/jobs/{slow['job_id']}").json()
    assert running["status"] == "running" and 0 < running["progress"] < 1
    assert client.get("/health").json()["queue"]["running"] == slow["job_id"]
    client.post(f"/jobs/{slow['job_id']}/cancel")
    assert wait_job(client, slow["job_id"])["status"] == "cancelled"
    assert wait_job(client, queued["job_id"])["status"] == "cancelled"
    # the queue keeps working after cancellations
    after = client.post("/txt2img", json={"prompt": "z", "backend": "mock", "steps": 1, "width": 256, "height": 256}).json()
    assert wait_job(client, after["job_id"])["status"] == "done"


def test_unload_endpoint(client):
    assert client.post("/unload").json() == {"ok": True}


def test_token(settings):
    from fastapi.testclient import TestClient

    from dronegen_server.app import create_app

    settings.token = "s3cret"
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 200  # health stays open: the add-on checks the connection
        body = {"prompt": "x", "backend": "mock", "steps": 1}
        assert c.post("/txt2img", json=body).status_code == 401
        r = c.post("/txt2img", json=body, headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200


def test_styles_are_listed_and_added_by_the_server(client):
    st = client.get("/styles").json()
    assert set(st) == {"flat", "render", "sheet"} and st["sheet"]["size"] == [1920, 576]
    r = client.post("/txt2img", json={"prompt": "домбра", "style": "sheet", "backend": "mock", "steps": 1})
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert (job["params"]["width"], job["params"]["height"]) == (1920, 576)  # the size of the style
    assert job["params"]["subject"] == "домбра" and "The object: домбра." in job["result"]["prompt"]
    meta = Image.open(io.BytesIO(client.get(f"/files/{job['files']['image']}").content)).info["dronegen"]
    assert '"style": "sheet"' in meta
    bad = client.post("/txt2img", json={"prompt": "x", "style": "oil", "backend": "mock"})
    assert bad.status_code == 400 and "стиль" in bad.json()["detail"]


def rgba_sheet():
    from PIL import ImageDraw

    img = Image.new("RGBA", (1100, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for x0, w, colour in ((80, 150, (150, 100, 50)), (380, 60, (150, 100, 50)), (600, 150, (40, 90, 160)),
                          (880, 60, (150, 100, 50))):  # front, side, back, the side again
        d.rectangle([x0, 40, x0 + w, 360], fill=(*colour, 255))
    return img


def test_sheet_split_into_cards_and_cards_used_for_3d(client, settings):
    rgba_sheet().save(settings.outputs / "sheet1.png")
    r = client.post("/cards/split", data={"image_name": "sheet1.png"})
    assert r.status_code == 200, r.text
    names = r.json()["cards"]
    assert list(names) == ["front", "side", "back"]
    assert r.json()["figures"] == ["front", "side", "back", "side again"]
    side = Image.open(io.BytesIO(client.get(f"/files/{names['side']}").content))
    assert side.mode == "RGBA" and side.width < 200
    # the front card is the picture for single-view methods
    r = client.post("/img2mesh", data={"backend": "silhouette", "mc_resolution": "64",
                                       "views": json.dumps({"front": names["front"], "side": names["side"]}),
                                       "params": json.dumps({"depth": 0.2})})
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["params"]["views"] == ["front", "side"] and job["params"]["depth"] == 0.2
    assert job["params"]["source"] == "front card"


def test_cards_as_uploads_and_bad_json(client):
    files = {"view_front": ("f.png", png_bytes(ring_image()), "image/png")}
    r = client.post("/img2mesh", files=files, data={"backend": "silhouette", "mc_resolution": "64", "remove_bg": "false"})
    assert wait_job(client, r.json()["job_id"])["status"] == "done"
    r = client.post("/img2mesh", data={"backend": "silhouette", "views": "{not json"})
    assert r.status_code == 400 and "views" in r.json()["detail"]
    r = client.post("/img2mesh", files=files, data={"backend": "silhouette", "params": "[1, 2]"})
    assert r.status_code == 400 and "params" in r.json()["detail"]
    r = client.post("/cards/split", data={"image_name": "missing.png"})
    assert r.status_code == 404
