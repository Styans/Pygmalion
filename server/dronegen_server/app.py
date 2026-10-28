"""HTTP API (FastAPI). All heavy work happens in the job worker, never in a request."""

from __future__ import annotations

import io
import ipaddress
import json
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, cards, effects, imaging, memory, parametric, rigging, styles
from .rigging import motions
from .chat_api import Studio
from .chat_api import router as studio_router
from .backends import Registry
from .config import Settings
from .errors import UserError, friendly
from .jobs import JobManager

log = logging.getLogger("dronegen.app")

FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


class Txt2ImgRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative: str = Field("", max_length=4000)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    steps: int | None = Field(None, ge=1, le=150)
    seed: int = -1
    backend: str = "zimage_turbo"
    # None: the prompt as is; flat | render | sheet: the prompt is the subject, styles.build adds the rest
    style: str | None = None


class BuildRequest(BaseModel):
    template: str
    values: dict[str, float] = Field(default_factory=dict)


class EffectRequest(BaseModel):
    kind: str
    size: float = Field(gt=0, description="metres across")
    spacing: float = Field(gt=0, description="the drones' safety distance, metres")
    frame: dict  # {"left", "up", "forward"}: the scene's axes of the figure
    reserve_at: list[float] = Field(min_length=3, max_length=3, description="the reserve's centre from the anchor")
    count: int = Field(0, ge=0, le=2000, description="0: as many as fit")
    params: dict = Field(default_factory=dict)
    points: list[list[float]] | None = None  # kind "points": the places (a trail, a model's surface)


class BakeRequest(BaseModel):
    skeleton: dict  # {"frame": {forward, left, up}, "size": float, "bones": [{name, role, side, chain, segment,
    #                  head, tail, parent, matrix (3×3, the bone's axes in the armature's space)}]}
    motion: str
    params: dict[str, float] = Field(default_factory=dict)
    samples: int = Field(24, ge=4, le=240)


def _npz_mesh(data: bytes):
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return np.asarray(z["vertices"]), np.asarray(z["faces"])
    except (KeyError, ValueError, OSError) as e:
        raise UserError(f"mesh: нужен NPZ с массивами vertices и faces ({e})") from None


def _glb_mesh(path: Path):
    """A GLB of the outputs in Blender axes, as Blender's glTF importer turns it: (x, y, z) -> (x, -z, y)."""
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    v = np.asarray(mesh.vertices, np.float64)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1), np.asarray(mesh.faces)


def _is_loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return host == "localhost"


class _GpuCache:
    """/health is polled often; querying CUDA each time is cheap but not free."""

    def __init__(self, ttl: float = 2.0):
        self.ttl, self.value, self.time, self.lock = ttl, None, 0.0, threading.Lock()

    def get(self) -> dict:
        with self.lock:
            if self.value is None or time.time() - self.time > self.ttl:
                self.value, self.time = memory.gpu_info(), time.time()
            return self.value


def create_app(settings: Settings, manager: JobManager | None = None) -> FastAPI:
    settings.ensure_dirs()
    manager = manager or JobManager(settings, Registry(settings))
    gpu_cache = _GpuCache()
    studio = Studio(settings, manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        log.info("DroneGen server %s ready, data in %s", __version__, settings.home)
        yield
        await studio.close()
        manager.stop()

    app = FastAPI(title="DroneGen", version=__version__, lifespan=lifespan)
    app.state.manager = manager
    app.state.settings = settings
    app.state.studio = studio

    def auth(request: Request) -> None:
        if not settings.token:
            return
        header = request.headers.get("authorization", "")
        query = request.query_params.get("token", "")  # EventSource cannot send headers
        if not (secrets.compare_digest(header, f"Bearer {settings.token}") or secrets.compare_digest(query, settings.token)):
            raise HTTPException(401, "Нужен токен доступа (настройки аддона → Токен)")

    @app.exception_handler(UserError)
    async def user_error(_request: Request, exc: UserError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_request: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors():
            where = ".".join(str(x) for x in err.get("loc", ())[1:]) or "запрос"
            parts.append(f"{where}: {err.get('msg', 'неверное значение')}")
        return JSONResponse({"detail": "Неверные параметры — " + "; ".join(parts)}, status_code=400)

    @app.exception_handler(Exception)
    async def internal(_request: Request, exc: Exception):
        log.exception("request failed")
        return JSONResponse({"detail": friendly(exc)}, status_code=500)

    @app.get("/health")
    def health():
        running = manager.running
        return {
            "name": "dronegen",
            "version": __version__,
            "backends": {
                kind: [b.describe() for b in manager.registry.of_kind(kind)] for kind in ("txt2img", "img2mesh", "build", "rig")
            },
            "gpu": gpu_cache.get(),
            "memory": memory.system_memory(),
            "queue": {"pending": manager.pending(), "running": running.id if running else None},
            "device": settings.device,
            "offline": settings.offline,
        }

    @app.get("/styles")
    def style_list():
        return {name: {**info, "size": list(info["size"])} for name, info in styles.STYLES.items()}

    @app.post("/txt2img", dependencies=[Depends(auth)])
    def txt2img(req: Txt2ImgRequest):
        params = req.model_dump(exclude={"backend", "style"})
        if req.style:
            if req.style not in styles.STYLES:
                raise UserError(f"Неизвестный стиль «{req.style}»: {', '.join(styles.STYLES)}")
            params["subject"] = req.prompt
            params["prompt"] = styles.build(req.prompt, req.style)
            params["style"] = req.style
            w, h = styles.STYLES[req.style]["size"]
            params["width"], params["height"] = params["width"] or w, params["height"] or h
        params["width"] = params["width"] or 768
        params["height"] = params["height"] or 768
        job = manager.submit("txt2img", req.backend, params)
        return {"job_id": job.id, "status": job.status, "queue_position": manager.queue_position(job)}

    upload_limit = settings.max_upload_mb * 1024 * 1024

    async def _upload(f: UploadFile) -> bytes:
        data = await f.read(upload_limit + 1)
        if len(data) > upload_limit:
            raise HTTPException(413, f"Файл больше {settings.max_upload_mb} МБ")
        return data

    @app.get("/templates")
    def templates():
        """The drawings of known objects and their parameters (name, range, default)."""
        return {
            key: {"title": t.title, "about": t.about,
                  "params": {n: {"default": p.default, "min": p.lo, "max": p.hi, "title": p.title, "unit": p.unit,
                                 "integer": p.integer} for n, p in t.params.items()}}
            for key, t in parametric.TEMPLATES.items()
        }

    @app.post("/build", dependencies=[Depends(auth)])
    def build(req: BuildRequest):
        job = manager.submit("build", "parametric", {"template": req.template, "values": req.values})
        return {"job_id": job.id, "status": job.status, "queue_position": manager.queue_position(job)}

    @app.post("/img2mesh", dependencies=[Depends(auth)])
    async def img2mesh(
        image: UploadFile | None = File(None),
        image_name: str | None = Form(None),
        backend: str = Form("triposr"),
        remove_bg: bool = Form(True),
        mc_resolution: int | None = Form(None),
        bg_method: str = Form("auto"),
        depth: float | None = Form(None),
        target_faces: int = Form(0),
        chunk_size: int | None = Form(None),
        foreground_ratio: float | None = Form(None),
        params: str | None = Form(None, description="JSON: any other settings of the backend"),
        views: str | None = Form(None, description='JSON {"front": "<file in outputs>", "side": ...}'),
        view_front: UploadFile | None = File(None),
        view_left: UploadFile | None = File(None),
        view_back: UploadFile | None = File(None),
        view_right: UploadFile | None = File(None),
        view_side: UploadFile | None = File(None),
    ):
        source, img = None, None
        if image is not None:
            img, source = imaging.open_image(await _upload(image)), image.filename or "upload"
        elif image_name:
            img, source = imaging.open_image(_output_file(image_name).read_bytes()), image_name
        cards_in: dict = {}
        try:
            named = json.loads(views) if views else {}
        except ValueError:
            raise UserError("views: нужен JSON вида {\"front\": \"имя файла\"}") from None
        for view, name in named.items():
            cards_in[view] = imaging.open_image(_output_file(str(name)).read_bytes())
        uploads = {"front": view_front, "left": view_left, "back": view_back, "right": view_right, "side": view_side}
        for view, f in uploads.items():
            if f is not None:
                cards_in[view] = imaging.open_image(await _upload(f))
        if img is None and "front" in cards_in:
            img, source = cards_in["front"], "front card"
        if img is None:
            raise UserError("Не передана картинка (поле image или image_name) или карточки видов")
        try:
            extra = json.loads(params) if params else {}
        except ValueError:
            raise UserError("params: нужен JSON с настройками") from None
        if not isinstance(extra, dict):
            raise UserError("params: нужен JSON-объект")
        job_params = {
            "remove_bg": remove_bg,
            "mc_resolution": mc_resolution,
            "bg_method": bg_method,
            "depth": depth,
            "target_faces": target_faces,
            "chunk_size": chunk_size,
            "foreground_ratio": foreground_ratio,
            **extra,
            "source": source,
        }
        inputs = {"image": img}
        if cards_in:
            inputs["views"] = cards_in
            job_params["views"] = sorted(cards_in)
        job = manager.submit("img2mesh", backend, job_params, inputs=inputs)
        return {"job_id": job.id, "status": job.status, "queue_position": manager.queue_position(job)}

    @app.get("/rig/plans")
    def rig_plans():
        """Body plans for the skeleton: name -> title and hints (how many legs, wings, arms; standing or not)."""
        return {name: {"title": rigging.PLAN_TITLES.get(name, name), "hints": hints}
                for name, hints in rigging.BODY_PLANS.items()}

    @app.post("/rig", dependencies=[Depends(auth)])
    async def rig(
        mesh: UploadFile | None = File(None, description="NPZ: vertices (N×3), faces (M×3) in Blender object space"),
        mesh_name: str | None = Form(None, description="a GLB from the outputs instead (glTF axes turned to Blender's)"),
        method: str = Form("template"),
        plan: str = Form("auto"),
    ):
        """A skeleton for a mesh: a job whose result holds the bones (``rigging.fit``)."""
        if mesh is not None:
            vertices, faces, source = *_npz_mesh(await _upload(mesh)), mesh.filename or "upload"
        elif mesh_name:
            vertices, faces, source = *_glb_mesh(_output_file(mesh_name)), mesh_name
        else:
            raise UserError("Не передана сетка (поле mesh — NPZ с vertices и faces — или mesh_name)")
        job = manager.submit("rig", method, {"plan": plan, "source": source, "vertices": len(vertices)},
                             inputs={"vertices": vertices, "faces": faces})
        return {"job_id": job.id, "status": job.status, "queue_position": manager.queue_position(job)}

    @app.get("/motions")
    def motion_list():
        """The motions for skeletons: name -> title, kind (cycle, pose, once), roles they need, period, parameters."""
        return motions.catalog()

    @app.post("/motion/bake", dependencies=[Depends(auth)])
    def motion_bake(req: BakeRequest):
        """Keys of a motion for a skeleton: local quaternions of the bones it turns and the root's shift.
        Milliseconds of arithmetic: answered at once, not queued."""
        return motions.bake(motions.skeleton(req.skeleton), req.motion, req.params, req.samples)

    @app.get("/effects")
    def effect_list():
        return effects.KINDS

    @app.post("/effects/build", dependencies=[Depends(auth)])
    def effect_build(req: EffectRequest):
        """Places of an effect's drones and their waiting places in the reserve, matched (answered at once)."""
        return effects.build(req.kind, req.size, req.spacing, req.frame, req.reserve_at, req.count, req.params,
                             req.points)

    @app.post("/cards/split", dependencies=[Depends(auth)])
    def split_cards(image_name: str = Form(...)):
        """A turnaround sheet from the outputs -> one card per view it shows (front, side, back; repeats left out)."""
        path = _output_file(image_name)
        sheet = cards.read_sheet(imaging.open_image(path.read_bytes()))
        out = {}
        for view, card in sheet.views.items():
            name = f"{path.stem}_card_{view}.png"
            imaging.save_png(card, settings.outputs / name, {"sheet": image_name, "view": view})
            out[view] = name
        return {"cards": out, "figures": sheet.roles, "notes": sheet.notes}

    @app.get("/jobs", dependencies=[Depends(auth)])
    def jobs(limit: int = 50):
        return [j.to_dict(manager.queue_position(j)) for j in manager.recent(limit)]

    @app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
    def job(job_id: str):
        j = manager.get(job_id)
        if j is None:
            raise HTTPException(404, "Задача не найдена (возможно, сервер перезапускали)")
        return j.to_dict(manager.queue_position(j))

    @app.post("/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
    def cancel(job_id: str):
        j = manager.cancel(job_id)
        if j is None:
            raise HTTPException(404, "Задача не найдена")
        return j.to_dict(manager.queue_position(j))

    def _output_file(name: str):
        if not FILE_NAME.match(name):
            raise HTTPException(404, "Файл не найден")
        root = settings.outputs.resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            raise HTTPException(404, "Файл не найден")
        return path

    @app.get("/files/{name}", dependencies=[Depends(auth)])
    def files(name: str):
        return FileResponse(_output_file(name), filename=name)

    @app.post("/unload", dependencies=[Depends(auth)])
    def unload():
        manager.request_unload()
        return {"ok": True}

    @app.post("/shutdown", dependencies=[Depends(auth)])
    def shutdown(request: Request):
        if not _is_loopback(request.client.host if request.client else None):
            raise HTTPException(403, "Остановить сервер можно только с этого же компьютера")
        server = getattr(app.state, "server", None)
        if server is None:
            raise HTTPException(409, "Сервер запущен не через python -m dronegen_server")
        server.should_exit = True
        return {"ok": True}

    app.include_router(studio_router(studio, auth))

    @app.get("/", include_in_schema=False)
    @app.get("/studio", include_in_schema=False)
    def studio_page():
        return RedirectResponse("/studio/")

    # the Studio web app (after the API routes, which take /studio/api first)
    app.mount("/studio", StaticFiles(directory=Path(__file__).with_name("studio"), html=True), name="studio")
    return app
