"""HTTP API of the Studio: projects, chats, messages, live events (SSE), settings, the Blender inbox.

Mounted by ``app.create_app`` under ``/studio/api``; the Blender add-on polls ``/blender/inbox``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import imaging, parametric
from .errors import UserError, friendly
from .jobs import ERROR, FINISHED
from .agents.assistant import ChatService, display
from .llm import Providers
from .llm import images as llm_images
from .llm.claude import MODELS
from .store import SECRETS, NotFound, Store, StudioConfig, new_id

log = logging.getLogger("dronegen.studio")

MAX_UPLOADS = 4


class ProjectIn(BaseModel):
    title: str = Field("Новый проект", max_length=200)


class ProjectPatch(BaseModel):
    title: str | None = Field(None, max_length=200)
    pinned: bool | None = None


class ChatIn(BaseModel):
    project_id: str
    title: str = Field("Новый чат", max_length=200)


class ChatPatch(BaseModel):
    title: str | None = Field(None, max_length=200)
    pinned: bool | None = None
    project_id: str | None = None
    precision: Literal["auto", "single", "sheet"] | None = Field(
        None, description="auto: сам; single: одна картинка (быстро); sheet: лист из 4 видов (точно)")


class SettingsIn(BaseModel):
    provider: str | None = None
    claude_model: str | None = None
    image_backend: str | None = None
    variants: int | None = Field(None, ge=1, le=4)
    web_lookup: bool | None = Field(None, description="искать в интернете (Википедия), как выглядят персонажи")
    anthropic_api_key: str | None = Field(None, max_length=500)
    sketchfab_token: str | None = Field(None, max_length=500)


class BlenderIn(BaseModel):
    file: str = Field(max_length=200)
    name: str = Field("Фигура", max_length=60)


class BuildIn(BaseModel):
    template: str
    values: dict[str, float] = Field(default_factory=dict)


class InboxDone(BaseModel):
    object: str = ""
    error: str = ""


class Studio:
    """Everything the Studio needs, created with the app and closed with it."""

    def __init__(self, settings, manager):
        folder = settings.home / "studio"
        self.settings = settings
        self.store = Store(folder)
        self.config = StudioConfig(folder)
        self.providers = Providers(settings, self.config, manager)
        self.chat = ChatService(settings, self.store, self.config, self.providers, manager)

    async def close(self) -> None:
        await self.chat.close()
        await self.providers.close()
        self.store.close()


def router(studio: Studio, auth) -> APIRouter:
    r = APIRouter(dependencies=[Depends(auth)])
    store, config, chat = studio.store, studio.config, studio.chat
    settings = studio.settings

    def found(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotFound as e:
            raise HTTPException(404, str(e)) from None

    # -- status and settings ------------------------------------------------------------
    @r.get("/studio/api/status")
    def status():
        return {"providers": studio.providers.status(), "settings": config.all(), "secrets": config.secrets_state(),
                "claude_models": list(MODELS)}

    @r.put("/studio/api/settings")
    def put_settings(body: SettingsIn):
        if body.provider is not None and body.provider not in ("local", "claude"):
            raise HTTPException(400, "provider: local или claude")
        if body.claude_model is not None and body.claude_model not in MODELS:
            raise HTTPException(400, "claude_model: " + ", ".join(MODELS))
        for key in SECRETS:
            value = getattr(body, key)
            if value is not None:
                config.set_secret(key, value)
        config.update(provider=body.provider, claude_model=body.claude_model, image_backend=body.image_backend,
                      variants=body.variants, web_lookup=body.web_lookup)
        return status()

    # -- projects -----------------------------------------------------------------------
    @r.get("/studio/api/projects")
    def projects():
        return store.projects()

    @r.post("/studio/api/projects")
    def add_project(body: ProjectIn):
        return store.add_project(body.title)

    @r.patch("/studio/api/projects/{project_id}")
    def patch_project(project_id: str, body: ProjectPatch):
        return found(store.update_project, project_id, body.title, body.pinned)

    @r.delete("/studio/api/projects/{project_id}")
    async def delete_project(project_id: str):
        found(store.project, project_id)
        for c in store.chats(project_id):
            await chat.stop(c["id"])
        store.delete_project(project_id)
        return {"ok": True}

    # -- chats --------------------------------------------------------------------------
    @r.get("/studio/api/chats")
    def chats(project_id: str | None = None, q: str = ""):
        return [{**c, "busy": chat.busy(c["id"])} for c in store.chats(project_id, q.strip())]

    @r.post("/studio/api/chats")
    def add_chat(body: ChatIn):
        c = found(store.add_chat, body.project_id, body.title)
        if body.title in ("", "Новый чат"):
            store.set_state(c["id"], auto_title=True)
        return found(store.chat, c["id"])

    @r.get("/studio/api/chats/{chat_id}")
    def get_chat(chat_id: str):
        c = found(store.chat, chat_id)
        return {
            "chat": c,
            "messages": [display(m) for m in store.messages(chat_id)],
            "assets": store.assets(chat_id),
            "busy": chat.busy(chat_id),
            "live": chat.live(chat_id),
            "pending": c["state"].get("pending"),
        }

    @r.patch("/studio/api/chats/{chat_id}")
    def patch_chat(chat_id: str, body: ChatPatch):
        updated = found(store.update_chat, chat_id, body.title, body.pinned, body.project_id)
        if body.precision is not None:
            store.set_state(chat_id, precision=None if body.precision == "auto" else body.precision)
            updated = store.chat(chat_id)
        return updated

    @r.delete("/studio/api/chats/{chat_id}")
    async def delete_chat(chat_id: str):
        found(store.chat, chat_id)
        await chat.stop(chat_id)
        store.delete_chat(chat_id)
        return {"ok": True}

    @r.post("/studio/api/chats/{chat_id}/messages", status_code=202)
    async def post_message(chat_id: str, text: str = Form(""), answers: str | None = Form(None),
                           views: str | None = Form(None, description="JSON: the side each file shows (front|left|back|right|'')"),
                           files: list[UploadFile] | None = File(None)):
        found(store.chat, chat_id)
        if chat.busy(chat_id):
            raise HTTPException(409, "Ассистент ещё отвечает — дождитесь или нажмите «Стоп»")
        try:
            parsed = json.loads(answers) if answers else None
        except ValueError:
            raise HTTPException(400, "answers: нужен JSON-список ответов") from None
        uploads = []
        limit = settings.max_upload_mb * 1024 * 1024
        for f in (files or [])[:MAX_UPLOADS]:
            data = await f.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"Файл больше {settings.max_upload_mb} МБ")
            img = imaging.open_image(data)
            name = f"{new_id()}_upload.png"
            imaging.save_png(img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB"),
                             settings.outputs / name, {"upload": f.filename or "", "chat": chat_id})
            uploads.append((img, name))
        if not text.strip() and not uploads and not parsed:
            raise HTTPException(400, "Пустое сообщение")
        try:
            sides = [str(v) for v in json.loads(views)] if views else []
        except ValueError:
            raise HTTPException(400, "views: нужен JSON-список видов") from None
        message = chat.user_message(text, uploads, parsed, sides)
        try:
            chat.send(chat_id, message)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        return {"ok": True}

    @r.post("/studio/api/chats/{chat_id}/blender")
    async def chat_to_blender(chat_id: str, body: BlenderIn):
        """The «В Blender» button: queue a model of this chat without asking the assistant."""
        found(store.chat, chat_id)
        path = (settings.outputs / body.file).resolve()
        if path.parent != settings.outputs.resolve() or not path.is_file() or path.suffix != ".glb":
            raise HTTPException(404, "Нет такой модели (.glb)")
        asset = store.asset_by_file(chat_id, body.file)
        item = store.add_inbox(body.file, body.name.strip() or "Фигура", chat_id, asset["id"] if asset else None)
        await chat.publish(chat_id, {"type": "inbox", "item": item})
        return item

    @r.post("/studio/api/chats/{chat_id}/build")
    async def chat_build(chat_id: str, body: BuildIn):
        """The sliders of a drawing: rebuild it with new parameter values, no model in the loop.

        While the last model of the chat is such an interactive build, it is replaced instead of
        adding a new one for every slider move.
        """
        found(store.chat, chat_id)
        try:
            job = chat.manager.submit("build", "parametric", {"template": body.template, "values": body.values})
        except UserError as e:
            raise HTTPException(400, str(e)) from None
        for _ in range(1200):  # up to 2 minutes (the queue may be busy with a picture)
            if job.status in FINISHED:
                break
            await asyncio.sleep(0.1)
        if job.status not in FINISHED:
            raise HTTPException(409, "Очередь занята — попробуйте позже")
        if job.status == ERROR:
            raise HTTPException(500, job.error or friendly(RuntimeError("сбой")))
        result = job.result
        data = {"method": "parametric", "template": body.template, "values": result["values"],
                "quality": result["quality"], "faces": result["faces"], "interactive": True}
        briefs = store.assets(chat_id, "brief")
        last = (store.assets(chat_id, "mesh") or [None])[-1]
        if last is not None and last["data"].get("interactive"):
            asset = store.update_asset(last["id"], data=data, files=dict(job.files))
        else:
            asset = store.add_asset(chat_id, "mesh", f"Чертёж: {result['title']}", dict(job.files), data,
                                    parent=briefs[-1]["id"] if briefs else None)
        return {"asset": asset["id"], "files": asset["files"], "quality": result["quality"], "values": result["values"],
                "template": body.template, "method": "parametric"}

    @r.get("/studio/api/templates")
    def templates():
        return {key: {"title": t.title, "about": t.about,
                      "params": {n: {"default": p.default, "min": p.lo, "max": p.hi, "title": p.title, "unit": p.unit,
                                     "integer": p.integer} for n, p in t.params.items()}}
                for key, t in parametric.TEMPLATES.items()}

    @r.post("/studio/api/chats/{chat_id}/stop")
    async def stop_chat(chat_id: str):
        found(store.chat, chat_id)
        return {"stopped": await chat.stop(chat_id)}

    async def sse(request: Request, key: str, hello: dict):
        q = chat.subscribe(key)

        def stopping() -> bool:
            server = getattr(request.app.state, "server", None)
            return bool(server is not None and server.should_exit)

        async def gen():
            try:
                yield f"event: hello\ndata: {json.dumps(hello, ensure_ascii=False)}\n\n"
                idle = 0
                while not stopping():  # an open stream must not hold the server back when it stops
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        idle += 1
                        if idle % 15 == 0:
                            if await request.is_disconnected():
                                break
                            yield ": keepalive\n\n"
                        continue
                    idle = 0
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            finally:
                chat.unsubscribe(key, q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @r.get("/studio/api/chats/{chat_id}/events")
    async def chat_events(chat_id: str, request: Request):
        found(store.chat, chat_id)
        return await sse(request, chat_id, {"busy": chat.busy(chat_id), "live": chat.live(chat_id)})

    @r.get("/studio/api/events")
    async def all_events(request: Request):
        """Every chat's events (the sidebar shows which chats are busy, titles, the Blender inbox)."""
        return await sse(request, "*", {"busy": [c for c in chat._tasks if chat.busy(c)]})

    @r.get("/studio/api/images/{name}")
    def chat_image(name: str):
        try:
            path = llm_images.path(studio.providers.image_dir, name)
        except ValueError:
            raise HTTPException(404, "Нет такой картинки") from None
        if not path.is_file():
            raise HTTPException(404, "Нет такой картинки")
        return FileResponse(path)

    # -- the Blender inbox ----------------------------------------------------------------
    @r.get("/blender/inbox")
    def inbox(status: str = "queued"):
        return store.inbox(status or None)

    async def _report(item: dict, text: str, level: str = "info"):
        if item.get("chat_id"):
            await chat.publish(item["chat_id"], {"type": "notice", "level": level, "text": text})
            await chat.publish(item["chat_id"], {"type": "inbox", "item": item})

    @r.post("/blender/inbox/{item_id}/taken")
    async def inbox_taken(item_id: str):
        item = found(store.update_inbox, item_id, "taken")
        await _report(item, f"Blender забирает модель «{item['name']}»…")
        return item

    @r.post("/blender/inbox/{item_id}/done")
    async def inbox_done(item_id: str, body: InboxDone):
        if body.error:
            item = found(store.update_inbox, item_id, "failed", body.error[:300])
            await _report(item, f"Blender не смог импортировать «{item['name']}»: {body.error}", "error")
        else:
            item = found(store.update_inbox, item_id, "imported", body.object[:200] or None)
            await _report(item, f"В Blender: объект «{item['object'] or item['name']}» в коллекции DroneGen")
        return item

    return r
