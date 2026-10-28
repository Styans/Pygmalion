"""What the chat assistant can do: questions, a look-up, a brief, pictures, a 3D model, sending it to Blender.

Every tool returns content for the model (text and pictures it looks at) and a ``ui`` payload for
the Studio (cards, galleries, the 3D preview). Long GPU jobs go through the usual job queue; their
progress is streamed to the chat. A tool that cannot go on without the user raises ``NeedsAnswer``:
the turn stops with its questions, as with ``ask_user``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from PIL import Image
from pydantic import BaseModel, Field

from .. import cards, imaging, parametric, preview, quality, styles
from ..errors import UserError, friendly
from ..jobs import CANCELLED, ERROR, FINISHED
from ..llm import Provider, Tool
from ..llm import images as llm_images
from ..store import new_id
from . import art_director, knowledge, lookup, prompt_engineer

log = logging.getLogger("dronegen.agents.tools")

OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
IMAGE_FALLBACKS = ("zimage_turbo", "sdxl")
VIEW_RU = {"front": "спереди", "left": "слева", "back": "сзади", "right": "справа"}


class ToolFailure(Exception):
    """A readable reason why a tool could not do its job (goes back to the model as an error)."""


class NeedsAnswer(Exception):
    """The tool cannot go on without the user. The turn stops with ``questions`` (as with ask_user) and the
    user's reply becomes this call's result after ``intro``. ``reference``: the reply describes how this
    thing looks and is kept as its reference. ``fallback``: the result when the turn already asks something."""

    def __init__(self, intro: str, questions: list[dict], ui: dict, fallback: str, reference: str = "", notice: str = ""):
        super().__init__(intro)
        self.intro, self.questions, self.ui, self.fallback = intro, questions, ui, fallback
        self.reference, self.notice = reference, notice


# -- inputs ------------------------------------------------------------------------------
class Question(BaseModel):
    question: str = Field(description="Короткий вопрос по-русски")
    options: list[str] = Field(description="2-5 коротких вариантов ответа", min_length=1, max_length=6)
    multiple: bool = Field(False, description="можно выбрать несколько вариантов")


class AskUser(BaseModel):
    questions: list[Question] = Field(description="1-3 вопроса", min_length=1, max_length=3)


class LookUp(BaseModel):
    name: str = Field(description="только название, как его написал пользователь, в именительном падеже: «Молния Маквин»",
                      json_schema_extra={"maxLength": 120})
    english: str = Field("", description="its usual English name if you know it (\"Lightning McQueen\"), else empty",
                         json_schema_extra={"maxLength": 120})


class WriteBrief(BaseModel):
    request: str = Field(description="Что сделать, по-русски, со всем, что сказал пользователь: предмет, поза, "
                                     "вид, цвета, детали, ответы на вопросы")


class GenerateImages(BaseModel):
    subject: str = Field(description="English description of the object: the brief's subject, corrected "
                                     "when a picture failed a check")
    count: int = Field(2, ge=1, le=4, description="how many variants, 1-4")
    seed: int | None = Field(None, description="optional; the same seed with a corrected subject keeps the composition")


class Photo(BaseModel):
    view: Literal["front", "left", "back", "right"] = Field(
        description="which side of the object the photo shows; left = the object's own left side")
    file: str = Field(description="file name of the photo, as in the user's message")


class Make3D(BaseModel):
    image: str = Field(description="file name of the chosen picture or view sheet, as generate_images returned it "
                                   "(or of the user's photo from the front)")
    photos: list[Photo] = Field(default_factory=list, description="ONLY when the user brought 2-4 photos of one "
                                "object from different sides: one entry per photo; the model is then built from them. "
                                "Photos the user marked with a side in the app are used automatically")


class Adjust(BaseModel):
    name: str = Field(description="parameter name from the drawing's table")
    value: float


TemplateKey = Literal[tuple(parametric.TEMPLATES)]  # type: ignore[valid-type]


class BuildExact(BaseModel):
    template: TemplateKey = Field(description="which exact drawing to build")
    adjust: list[Adjust] = Field(default_factory=list, description="ONLY parameters the user asked to change "
                                 "(longer neck, wider body...), names and ranges as in the drawing's table")


class SendToBlender(BaseModel):
    mesh: str = Field(description="file name of the .glb model, as make_3d returned it")
    name: str = Field(description="object name in Blender, short")


TOOLS = [
    Tool("ask_user",
         "Ask the user 1-3 short clarifying questions with answer options (buttons in the chat). Use it when "
         "the answer changes the result (style, pose, view, colours, level of detail) or to let the user pick "
         "a picture variant. Call it once, as the last call of your answer: the turn ends and the answer "
         "comes as the next message.", AskUser),
    Tool("look_up",
         "Find out what a named character or thing looks like: a short description and a picture from Wikipedia. "
         "Call it BEFORE write_brief whenever the request names something specific: a character of a cartoon, film, "
         "game or book, a brand or logo, a famous building, a real person (Молния Маквин, Губка Боб, логотип Apple, "
         "Эйфелева башня), even if you think you know it. Not needed for ordinary things (a car, an eagle, a heart) "
         "and the traditional Kazakh objects. The reference it finds goes into the brief by itself. Without the "
         "internet it asks the user to describe the thing or attach a picture.", LookUp),
    Tool("write_brief",
         "Have the prompt engineer write the brief for a figure: a precise English description, the picture "
         "style, the 3D method and the checks every picture must pass. Call it before the first picture of a "
         "figure and whenever the user changes what the figure should be (or asks for another style).", WriteBrief),
    Tool("generate_images",
         "Draw pictures from the latest brief (style and view come from it). Returns the pictures and the "
         "program's count of their parts: look at each picture and compare it with the brief's checks before "
         "calling it good.", GenerateImages),
    Tool("make_3d",
         "Make the 3D model from a chosen picture or view sheet; the method and the symmetry come from the "
         "brief. Returns a preview of the model (front, three-quarter, side) and shape measurements to check.", Make3D),
    Tool("build_exact",
         "Build the EXACT model of a known object from its drawing (no neural network, no pictures, instant): "
         "use it when the brief has a template, or when the user asks for such an object. Exactly the right "
         "parts and counts (a dombra has two pegs). Returns a preview and the drawing's parameter table.", BuildExact),
    Tool("send_to_blender",
         "Send a finished model to Blender: the DroneGen add-on in the open Blender imports it into the DroneGen "
         "collection. Call it when the user agrees or asks for it.", SendToBlender),
]
BY_NAME = {t.name: t for t in TOOLS}
TITLES = {"ask_user": "Вопрос", "look_up": "Справка", "write_brief": "Бриф", "generate_images": "Картинки",
          "make_3d": "3D-модель", "build_exact": "Точная модель", "send_to_blender": "В Blender"}


@dataclass
class ToolContext:
    chat_id: str
    settings: object
    store: object
    config: object
    manager: object
    provider: Provider
    image_dir: Path
    publish: Callable[[dict], Awaitable[None]]


# -- helpers -----------------------------------------------------------------------------
FILE_IN_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|jpe?g|webp|glb|obj)", re.IGNORECASE)
# a picture chosen by typing («Вариант 1», «2»): not a request (a short one like «лошадь» is)
PICK = re.compile(r"^(?:вариант|картинк[аиу]|номер|№)?\s*\d+\s*[.!)]?$", re.IGNORECASE)


def clean_name(name: str) -> str:
    """A small model copies the words around a file name too («файл 1a2b_upload.png»): keep the name itself."""
    name = (name or "").strip().strip("«»\"'`")
    if OUTPUT_NAME.match(name):
        return name
    found = FILE_IN_TEXT.findall(name)
    return found[-1] if len(found) == 1 else name


def output_file(ctx: ToolContext, name: str) -> Path:
    name = clean_name(name)
    if not OUTPUT_NAME.match(name or ""):
        raise ToolFailure(f"«{name}» — не имя файла из этого чата")
    path = (ctx.settings.outputs / name).resolve()
    if path.parent != ctx.settings.outputs.resolve() or not path.is_file():
        raise ToolFailure(f"Файл {name} не найден: используйте имя из результата generate_images или make_3d")
    return path


async def run_job(ctx: ToolContext, use_id: str, job, label: str):
    """Wait for a queued job, streaming its progress; the job is cancelled with the turn."""
    last = None
    try:
        while job.status not in FINISHED:
            state = (round(job.progress, 2), job.stage)
            if state != last:
                last = state
                await ctx.publish({"type": "tool_progress", "tool_use_id": use_id, "label": label,
                                   "progress": job.progress, "stage": job.stage,
                                   "queue": ctx.manager.queue_position(job)})
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        ctx.manager.cancel(job.id)
        raise
    if job.status == ERROR:
        raise ToolFailure(job.error or "задача не удалась")
    if job.status == CANCELLED:
        raise ToolFailure("задача отменена")
    return job


def submit(ctx: ToolContext, kind: str, backend: str, params: dict, inputs: dict | None = None):
    b = ctx.manager.registry.get(kind, backend)
    ok, reason = b.check()
    if not ok:
        raise ToolFailure(f"{b.title} недоступен: {reason}")
    return ctx.manager.submit(kind, backend, params, inputs)


def image_backend(ctx: ToolContext) -> str:
    wanted = ctx.config.get("image_backend")
    for name in (wanted, *IMAGE_FALLBACKS):
        try:
            if ctx.manager.registry.get("txt2img", name).check()[0]:
                return name
        except UserError:
            continue
    raise ToolFailure("Нет доступной модели для картинок (Z-Image или SDXL): запустите python install.py")


def latest(ctx: ToolContext, kind: str) -> dict | None:
    found = ctx.store.assets(ctx.chat_id, kind)
    return found[-1] if found else None


def user_words(ctx: ToolContext) -> list[str]:
    """What the user themself wrote in this chat, oldest first: messages and their answers to questions."""
    out = []
    for m in ctx.store.messages(ctx.chat_id):
        if m["role"] != "user":
            continue
        for b in m["content"]:
            if b["type"] == "text":
                out.append(b["text"])
            elif b["type"] == "tool_result" and b.get("ui", {}).get("kind") == "answer":
                out.extend(c["text"] for c in b["content"] if c["type"] == "text")
    return out


# -- tools -------------------------------------------------------------------------------
LOOK_OPTIONS = ("Опишу словами", "Прикреплю картинку", "Делай как понимаешь")


def ask_how_it_looks(name: str, why: str, notice: str = "") -> NeedsAnswer:
    """The look-up failed: the user describes the thing or attaches a picture (or lets the model guess).
    The first two options only point at the input field and the paper clip in the Studio (``actions``)."""
    question = {"question": f"Не нашёл, как выглядит «{name}»: {why}. Опишите словами или прикрепите картинку.",
                "options": list(LOOK_OPTIONS), "multiple": False,
                "actions": {LOOK_OPTIONS[0]: "describe", LOOK_OPTIONS[1]: "attach"}}
    intro = f"Справку о «{name}» найти не удалось ({why}), и пользователя спросили, как это выглядит. Ответ пользователя:"
    fallback = (f"Справку о «{name}» найти не удалось: {why}. Спроси пользователя через ask_user, как это выглядит: "
                "пусть опишет словами или прикрепит картинку.")
    return NeedsAnswer(intro, [question], {"kind": "lookup", "name": name, "why": why}, fallback, name, notice)


def reference_result(asset: dict):
    """What the model and the Studio get for a reference: its text, its picture and where it came from."""
    d = asset["data"]
    where = "из Википедии" if d.get("source") == "wikipedia" else "от пользователя"
    text = f"Справка {where} о «{d['title']}»: {d.get('text') or '(только картинка)'}"
    if d.get("text_en"):
        text += f"\nEnglish: {d['text_en']}"
    text += "\nКартинка ниже: посмотри, как это выглядит." if d.get("picture") else "\nКартинки нет."
    text += "\nСправка сохранена и сама попадёт в бриф: дальше вызывай write_brief."
    content = [{"type": "text", "text": text}]
    if d.get("picture"):
        content.append({"type": "image", "file": d["picture"]})
    return content, {"kind": "reference", "asset": asset["id"], "name": d.get("name", ""), "title": d["title"],
                     "text": d.get("text", ""), "url": d.get("url", ""), "image": asset["files"].get("image"),
                     "source": d.get("source", "")}


def references_for(ctx: ToolContext, request: str) -> list[dict]:
    """The references a new brief uses: found or described since the last brief, or named in the request
    (an old reference must not leak into the brief of another figure)."""
    briefs = ctx.store.assets(ctx.chat_id, "brief")
    since = briefs[-1]["created"] if briefs else 0
    refs = [r for r in ctx.store.assets(ctx.chat_id, "reference")
            if r["created"] > since or lookup.score(r["data"].get("name", ""), request) > 0]
    return refs[-2:]


async def look_up(ctx: ToolContext, a: LookUp, use_id: str):
    name = a.name.strip().strip("«»\"'").strip()
    if not name:
        raise ToolFailure("Не указано, что искать: передай название в name")
    if knowledge.match(name):  # the traditional objects have their own reference sheets
        text = f"«{name}» есть во встроенной справке: искать не нужно, вызывай write_brief."
        return [{"type": "text", "text": text}], {"kind": "reference", "known": True, "name": name, "title": name}
    for ref in reversed(ctx.store.assets(ctx.chat_id, "reference")):  # already known in this chat
        if lookup.same(ref["data"].get("name", ""), name) or lookup.same(ref["data"].get("title", ""), name):
            return reference_result(ref)
    if not ctx.config.get("web_lookup"):
        raise ask_how_it_looks(name, "поиск в интернете выключен в настройках студии")
    await ctx.publish({"type": "tool_progress", "tool_use_id": use_id, "label": "Википедия", "progress": 0.3,
                       "stage": f"ищу «{name}»"})
    try:
        found = await lookup.find(name, a.english)
    except lookup.Offline:
        raise ask_how_it_looks(name, "нет связи с интернетом", f"Нет интернета: не могу узнать, как выглядит «{name}». "
                               "Опишите словами или прикрепите картинку.") from None
    except lookup.Unavailable as e:
        raise ask_how_it_looks(name, str(e), f"Не удалось найти «{name}» в интернете: {e}.") from None
    if found is None:
        raise ask_how_it_looks(name, "в Википедии такой статьи нет")
    files, picture = {}, ""
    if found.image is not None:
        files["image"] = f"{new_id()}_reference.png"
        imaging.save_png(found.image, ctx.settings.outputs / files["image"], {"reference": found.url})
        picture = llm_images.add(ctx.image_dir, found.image)
    asset = ctx.store.add_asset(ctx.chat_id, "reference", found.title, files,
                                {"name": name, "title": found.title, "url": found.url, "text": found.text,
                                 "text_en": found.text_en, "source": "wikipedia", "picture": picture})
    return reference_result(asset)


async def write_brief(ctx: ToolContext, a: WriteBrief, use_id: str):
    await ctx.publish({"type": "tool_progress", "tool_use_id": use_id, "label": "промпт-инженер", "progress": 0.1,
                       "stage": "пишу точное описание"})
    precision = ctx.store.chat(ctx.chat_id)["state"].get("precision")  # the «Точность» switch of the chat
    style = prompt_engineer.style_hint(user_words(ctx), precision)
    if calls_this_turn(ctx, "write_brief") > 3:  # a small model can rewrite the brief for ever
        raise ToolFailure("Бриф уже переписан несколько раз. Не вызывай write_brief снова: используй последний бриф "
                          "или спроси пользователя, что изменить")
    # the user's own words (not the assistant's paraphrase with invented checks like "1 крыло", not "Вариант 1")
    briefs = ctx.store.assets(ctx.chat_id, "brief")
    said = [w.strip() for w in typed_words(ctx, briefs[-1]["created"] if briefs else 0)]
    said = [w for w in said if w and not PICK.match(w)][-3:]
    last = briefs[-1] if briefs else None
    before = (last["data"].get("request") or last["title"]) if last else ""
    if last and said and not await prompt_engineer.same_figure(ctx.provider, last["title"] or before[:60], said):
        before = ""  # another figure: its brief is written from the new words alone, as in a new chat (carried into
        # the eagle's request with the eagle's reference sheet, «мне нужна лошадь» gave an eagle again)
    if before:  # the same figure: what it is comes from the last brief, not from words like «проблемы с текстурой»
        # typed since (they once were the whole request, and the eagle was lost)
        request = (f"{before[:400]}\nПотом пользователь написал: " + " ".join(said) if said else before)[:700]
    else:
        request = " ".join(said)[:700] if said else a.request
    refs = references_for(ctx, request)
    brief = await prompt_engineer.write_brief(ctx.provider, request, style, references=[r["data"] for r in refs])
    data = {**brief.model_dump(), "prompt": brief.prompt(), "request": request}
    if refs:
        data["references"] = [r["data"]["title"] for r in refs]
    asset = ctx.store.add_asset(ctx.chat_id, "brief", brief.title_ru, {}, data)
    chat = ctx.store.chat(ctx.chat_id)
    if chat["title"] in ("Новый чат", "") or chat["state"].get("auto_title"):
        chat = ctx.store.update_chat(ctx.chat_id, title=brief.title_ru)
        ctx.store.set_state(ctx.chat_id, auto_title=None)
        await ctx.publish({"type": "chat", "chat": {"id": chat["id"], "title": chat["title"]}})
    text = "Бриф:\n" + json.dumps(brief.model_dump(), ensure_ascii=False, indent=1)
    if refs:
        text += "\n\nБриф написан по справке: " + ", ".join(f"«{t}»" for t in data["references"])
    known = [e for e in knowledge.match(request) if e.template]
    if len(known) == 1:  # an exact drawing exists: remember it in the brief and tell the model
        data["template"] = known[0].template
        data["notes_ru"] = ("Для этого предмета есть точный чертёж: строю по нему (мгновенно, без нейросети, с нужным "
                            "числом деталей). Нарисовать нейросетью можно по запросу.")
        asset = ctx.store.update_asset(asset["id"], data=data)
        text += (f"\n\nЕСТЬ ТОЧНЫЙ ЧЕРТЁЖ «{parametric.get(known[0].template).title}» ({known[0].template}): "
                 "построй его инструментом build_exact. Он точнее и быстрее нейросети (нужное число деталей, "
                 "ровные формы). Рисуй картинки, только если пользователь просит именно рисунок.")
    return [{"type": "text", "text": text}], {"kind": "brief", "asset": asset["id"], **data}


async def generate_images(ctx: ToolContext, a: GenerateImages, use_id: str):
    brief = latest(ctx, "brief")
    if brief is None:
        raise ToolFailure("Сначала вызови write_brief: картинки рисуются по брифу")
    backend = image_backend(ctx)
    style, view = brief["data"]["style"], brief["data"]["view"]  # the brief decides, not the model
    prompt = styles.build(a.subject, style, view)
    width, height = styles.STYLES[style]["size"]
    seed0 = a.seed if a.seed is not None else random.randint(0, 2**31 - 1 - a.count)
    drawn = {(x["data"].get("prompt"), x["data"].get("seed")) for x in ctx.store.assets(ctx.chat_id, "image")}
    if any((prompt, seed0 + i) in drawn for i in range(a.count)):  # the same prompt and seed draw the same picture
        seed0 = random.randint(0, 2**31 - 1 - a.count)
    jobs = []
    for i in range(a.count):
        params = {"prompt": prompt, "width": width, "height": height, "seed": seed0 + i, "subject": a.subject,
                  "style": style}
        if backend == "sdxl":
            params["negative"] = (brief or {}).get("data", {}).get("negative") or styles.NEGATIVE
        jobs.append(submit(ctx, "txt2img", backend, params))
    content, items, to_check = [{"type": "text", "text": f"Полный промпт: {prompt}"}], [], []
    try:
        for i, job in enumerate(jobs):
            job = await run_job(ctx, use_id, job, f"картинка {i + 1} из {a.count}")
            name = job.files["image"]
            img = Image.open(ctx.settings.outputs / name)
            img.load()
            item = {"n": i + 1, "file": name, "seed": job.result.get("seed"), "style": style}
            note, front = "", img
            if style == "sheet":
                await ctx.publish({"type": "tool_progress", "tool_use_id": use_id, "label": f"карточки {i + 1}",
                                   "progress": 0.95, "stage": "режу лист на карточки"})
                try:
                    sheet = await asyncio.to_thread(cards.read_sheet, img)
                    item["cards"] = {}
                    for view, card in sheet.views.items():
                        card_name = f"{Path(name).stem}_card_{view}.png"
                        imaging.save_png(card, ctx.settings.outputs / card_name, {"sheet": name, "view": view})
                        item["cards"][view] = card_name
                    if sheet.notes:
                        item["cards_note"] = sheet.report()
                    front = cards.square(sheet.views["front"], 768)  # the front card shows the details to count
                    note = f"; лист разрезан. {sheet.report()}"
                    if "side" in sheet.views:  # repeats are normal for the image model: a redraw draws them again
                        note += ". Для 3D этого хватает, из-за повторов не перерисовывай"
                    else:
                        note += (". Без вида сбоку 3D будет по одному виду спереди — менее точно; можно предложить "
                                 "перерисовать лист")
                except UserError as e:
                    item["cards_error"] = f"Лист не разрезан на карточки: {e}"
                    note = f"; лист НЕ разрезан на карточки ({e}): 3D по нему не сделать, нужен другой вариант"
            asset = ctx.store.add_asset(ctx.chat_id, "image", f"Вариант {i + 1}", {"image": name, **item.get("cards", {})},
                                        {"prompt": prompt, "subject": a.subject, "style": style, "view": view,
                                         "seed": item["seed"], "backend": backend, "cards": item.get("cards")},
                                        parent=brief["id"] if brief else None)
            item["asset"] = asset["id"]
            items.append(item)
            to_check.append((item, front))
            content.append({"type": "text", "text": f"Вариант {i + 1}: файл {name}, seed {item['seed']}{note}"})
            content.append({"type": "image", "file": llm_images.add(ctx.image_dir, img)})
            await ctx.publish({"type": "tool_item", "tool_use_id": use_id, "ui": {"kind": "image", **item}})
    except BaseException:
        for job in jobs:
            ctx.manager.cancel(job.id)
        raise
    # counting one thing (a beak, a head) only checks that it is there, and a small model miscounts it often:
    # the art director counts what there are several of (pegs, legs, wings, wheels)
    parts = [(c["part"], c["expected"], c.get("where", "whole")) for c in (brief or {}).get("data", {}).get("counts") or []
             if int(c.get("expected", 0)) >= 2]
    # one automatic redraw at most: then the user sees the variants with the marks and decides (a small model can
    # loop here for ever, and the counting itself is not always right)
    attempt = calls_this_turn(ctx, "generate_images")
    if parts and attempt >= 3:
        content.append({"type": "text", "text": "Лимит перерисовок исчерпан, проверку деталей пропускаю: покажи варианты "
                                                "пользователю как есть и дай выбрать."})
        parts = []
    if parts:  # the art director counts the parts on each picture on its own
        for k, (item, front) in enumerate(to_check):
            await ctx.publish({"type": "tool_progress", "tool_use_id": use_id, "label": "арт-директор",
                               "progress": (k + 0.5) / len(to_check), "stage": f"считаю детали, вариант {item['n']}"})
            verdict = await art_director.check(ctx.provider, ctx.image_dir, llm_images.add(ctx.image_dir, front), parts)
            item["check"] = verdict.model_dump() if verdict else None
            content.append({"type": "text", "text": f"Проверка варианта {item['n']}: {art_director.summary(verdict)}"})
            await ctx.publish({"type": "tool_item", "tool_use_id": use_id, "ui": {"kind": "image", **item}})
        if attempt >= 2:
            content.append({"type": "text", "text": "Это уже перерисовка: больше не перерисовывай. Покажи варианты "
                                                    "пользователю (с результатами проверки) и спроси через ask_user, какой брать."})
    return content, {"kind": "images", "items": items, "prompt": prompt, "backend": backend}


def typed_words(ctx: ToolContext, since: float = 0) -> list[str]:
    """Only what the user typed themself (after ``since``), oldest first: no answers to questions ("Вариант 1"), no
    notes about attached pictures. The brief is written from this."""
    out = []
    for m in ctx.store.messages(ctx.chat_id):
        if m["role"] != "user" or m["created"] <= since:
            continue
        for b in m["content"]:
            if b["type"] == "text" and not b["text"].startswith(("(картинка пользователя", "Ответы:")):
                out.append(b["text"])
    return out


def calls_this_turn(ctx: ToolContext, tool: str) -> int:
    """How many times ``tool`` was already called since the user's last message."""
    n = 0
    for m in reversed(ctx.store.messages(ctx.chat_id)):
        if m["role"] == "user" and any(b["type"] == "text" for b in m["content"]):
            break
        if m["role"] == "assistant":
            n += sum(1 for b in m["content"] if b.get("type") == "tool_use" and b.get("name") == tool)
    return n


def labelled_photos(ctx: ToolContext) -> dict[str, str]:
    """{view: file} of the photos the user attached to their latest message and marked with a side
    (the picker next to the picture in the composer): their choice, not the model's guess."""
    for m in reversed(ctx.store.messages(ctx.chat_id)):
        if m["role"] != "user" or not any(b.get("type") in ("text", "image") for b in m["content"]):
            continue  # tool results and answers are not the user's new message
        # only the newest message counts: photos sent earlier must not replace a picture made later
        return {b["view"]: b["source"] for b in m["content"] if b.get("type") == "image" and b.get("view") and b.get("source")}
    return {}


def user_uploads(ctx: ToolContext) -> set[str]:
    """Files of the pictures the user attached in this chat (drawn pictures and look-up references are not photos)."""
    return {b["source"] for m in ctx.store.messages(ctx.chat_id) if m["role"] == "user"
            for b in m["content"] if b.get("type") == "image" and b.get("source")}


def model_photos(ctx: ToolContext, photos: list[Photo]) -> dict[str, str]:
    """{view: file} of the photos the model passed. A small model fills ``photos`` with the drawn picture repeated
    for every side: the «multi-view» model then gets four fronts and builds nonsense, its colours a face on the back."""
    uploads, out = user_uploads(ctx), {}
    for photo in photos:
        file = clean_name(photo.file)
        if photo.view in out:
            raise ToolFailure(f"Два фото для вида «{photo.view}»: у каждого вида одно фото")
        if file not in uploads:
            raise ToolFailure(f"{file} — не фото пользователя. photos — только для фото, которые пользователь сам прислал "
                              "с разных сторон. Картинку из generate_images передавай в image, а photos не заполняй")
        if file in out.values():
            raise ToolFailure(f"Фото {file} указано для нескольких видов: у каждого вида должно быть своё фото. "
                              "Одна картинка — это один вид, остальные стороны так не получить")
        out[photo.view] = file
    return out


def method_for(ctx: ToolContext, source: dict | None) -> tuple[str, bool]:
    """(3D method, symmetry) for a picture: from the style it was drawn in and from the brief.
    A picture the user brought themself is one shaded picture."""
    brief = (latest(ctx, "brief") or {}).get("data", {})
    style = (source or {}).get("data", {}).get("style")
    if style not in prompt_engineer.METHOD_OF:
        return "hunyuan", bool(brief.get("symmetry"))
    return prompt_engineer.METHOD_OF[style], bool(brief.get("symmetry"))


async def make_3d(ctx: ToolContext, a: Make3D, use_id: str):
    fallback_note = ""
    photo_files = labelled_photos(ctx)
    if photo_files:  # the photos the user marked: the front one is the picture, whatever the model passed
        a = a.model_copy(update={"image": photo_files.get("front") or a.image})
    a = a.model_copy(update={"image": clean_name(a.image)})
    path = output_file(ctx, a.image)
    img = Image.open(path)
    img.load()
    source = ctx.store.asset_by_file(ctx.chat_id, a.image)
    method, symmetry = method_for(ctx, source)  # the brief decides, not the model
    inputs, params = {"image": img}, {}
    surface = prompt_engineer.surface_hint(user_words(ctx))  # from the user's words, never the model's whim
    if not photo_files:
        photo_files = model_photos(ctx, a.photos)
    if photo_files:  # the user's own photos from several sides: the most exact input there is
        method = "hunyuan_mv"
        views = {}
        for view, file in photo_files.items():
            card = Image.open(output_file(ctx, file))
            card.load()
            views[view] = card
        if "front" not in views:
            if a.image in photo_files.values():
                raise ToolFailure("Среди фото нет вида спереди: попроси пользователя отметить, какое фото спереди")
            views["front"] = img
        inputs["views"] = views
        try:  # the sheet only shows the user what goes into the model: a failure here must not stop the 3D
            await ctx.publish({"type": "tool_progress", "tool_use_id": use_id, "label": "лист из фото", "progress": 0.05,
                               "stage": f"склеиваю {len(views)} фото в один лист"})
            cut = {v: (await asyncio.to_thread(imaging.remove_background, c, "auto"))[0] for v, c in views.items()}
            sheet_name = f"{Path(a.image).stem}_photos_sheet.png"
            imaging.save_png(await asyncio.to_thread(cards.compose_sheet, cut), ctx.settings.outputs / sheet_name,
                             {"sheet_of_photos": sorted(views)})
            source = ctx.store.add_asset(ctx.chat_id, "image", "Лист из ваших фото", {"image": sheet_name},
                                         {"style": "sheet", "views": sorted(views)})
            await ctx.publish({"type": "tool_item", "tool_use_id": use_id,
                               "ui": {"kind": "image", "n": 1, "file": sheet_name, "asset": source["id"], "style": "sheet"}})
        except Exception as e:  # noqa: BLE001
            log.warning("лист из фото не собран: %s", e)
    if method.startswith("hunyuan"):
        params = {"symmetry": symmetry, "style": surface}
    if method == "hunyuan_mv" and "views" not in inputs:
        # read anew rather than taking the saved cards: sheets cut before views were told apart by what they show
        # have a repeated profile saved as «right» or «back»
        sheet_img = img
        sheet_file = (source or {}).get("files", {}).get("image")
        if sheet_file and sheet_file != a.image:
            try:
                sheet_img = Image.open(output_file(ctx, sheet_file))
                sheet_img.load()
            except ToolFailure:
                pass
        sheet = await asyncio.to_thread(cards.read_sheet, sheet_img)
        if len(sheet.views) >= 2:
            inputs["views"] = sheet.views
            if "side" not in sheet.views:
                fallback_note = ("На листе нет вида сбоку: модель построена по видам спереди и сзади "
                                 "(сбоку форма достроена).")
            elif "back" not in sheet.views:
                fallback_note = ("На листе нет вида сзади: модель построена по видам спереди и сбоку "
                                 "(сзади форма достроена).")
        else:
            method = "hunyuan"
            inputs["image"] = sheet.views["front"]
            fallback_note = (f"На листе нет вида сбоку: модель построена по виду спереди ({sheet.report()}). "
                             "Для точности со всех сторон можно нарисовать лист заново.")
    # the same picture with the same settings gives the same shape: a small model «fixes» a complaint this way
    repeat = next((m for m in reversed(ctx.store.assets(ctx.chat_id, "mesh"))
                   if m["data"].get("source") == a.image and m["data"].get("method") == method
                   and m["data"].get("symmetry") == symmetry and m["data"].get("surface") == surface
                   and (m["data"].get("photos") or {}) == photo_files), None)
    job = await run_job(ctx, use_id, submit(ctx, "img2mesh", method, params, inputs), "3D-модель")
    files = dict(job.files)
    import trimesh

    mesh = await asyncio.to_thread(trimesh.load, ctx.settings.outputs / files["mesh"], force="mesh")
    if "preview" not in files:
        name = f"{job.id}_preview.png"
        sheet = await asyncio.to_thread(preview.sheet, mesh)
        sheet.save(ctx.settings.outputs / name)
        files["preview"] = name
    q = job.result.get("quality")
    if not q:
        front = imaging.remove_background(img, "auto")[0] if method != "hunyuan_mv" else None
        q = await asyncio.to_thread(quality.report, mesh, front)
    asset = ctx.store.add_asset(ctx.chat_id, "mesh", f"3D: {Path(a.image).stem}", files,
                                {"method": method, "quality": q, "faces": len(mesh.faces), "source": a.image,
                                 "symmetry": symmetry, "surface": surface, "side_note": job.result.get("side_note"),
                                 "photos": photo_files},
                                parent=source["id"] if source else None)
    parts = [f"Модель готова: файл {files['mesh']}, {len(mesh.faces)} треугольников",
             f"толщина/ширина {q['depth_ratio']:.2f}"]
    if q.get("front_iou") is not None:
        parts.append(f"совпадение силуэта спереди с картинкой {q['front_iou']:.2f}")
    if job.result.get("side_note"):
        parts.append(job.result["side_note"])
    seen = [v for v in job.result.get("views") or () if v in VIEW_RU]
    if seen:  # what the colours are made of: the model must not promise «exact colours» from one picture
        unseen = [VIEW_RU[v] for v in VIEW_RU if v not in seen]
        parts.append("цвета взяты с видов: " + ", ".join(VIEW_RU[v] for v in seen)
                     + (f"; {', '.join(unseen)} на картинках нет — там цвета достроены приблизительно, без деталей. "
                        "Точные цвета со всех сторон даёт лист из 4 видов" if unseen else ""))
    if repeat is not None:
        parts.append(f"Это повтор: из этой картинки с теми же настройками уже сделана модель {repeat['files'].get('mesh')}, "
                     "форма та же. Не говори пользователю, что что-то исправлено")
    if fallback_note:
        parts.append(fallback_note)
        await ctx.publish({"type": "notice", "level": "warn", "text": fallback_note})
    if job.warnings:
        parts.append("предупреждения: " + "; ".join(job.warnings))
    preview_img = Image.open(ctx.settings.outputs / files["preview"])
    content = [{"type": "text", "text": ". ".join(parts) + ". Превью: спереди, 3/4, сбоку."},
               {"type": "image", "file": llm_images.add(ctx.image_dir, preview_img)}]
    return content, {"kind": "mesh", "asset": asset["id"], "files": files, "quality": q, "method": method,
                     "warnings": list(job.warnings)}


async def send_to_blender(ctx: ToolContext, a: SendToBlender, use_id: str):
    a = a.model_copy(update={"mesh": clean_name(a.mesh)})
    output_file(ctx, a.mesh)
    if not a.mesh.endswith(".glb"):
        raise ToolFailure("В Blender отправляется файл .glb из результата make_3d")
    source = ctx.store.asset_by_file(ctx.chat_id, a.mesh)
    item = ctx.store.add_inbox(a.mesh, a.name.strip()[:60] or "Фигура", ctx.chat_id, source["id"] if source else None)
    await ctx.publish({"type": "inbox", "item": item})
    text = (f"Модель «{item['name']}» поставлена в очередь для Blender. Аддон DroneGen в открытом Blender "
            "заберёт её и добавит в коллекцию DroneGen.")
    return [{"type": "text", "text": text}], {"kind": "blender", "inbox": item["id"], "name": item["name"], "file": a.mesh}


async def build_exact(ctx: ToolContext, a: BuildExact, use_id: str):
    changes = {adj.name: adj.value for adj in a.adjust}
    job = await run_job(ctx, use_id, submit(ctx, "build", "parametric", {"template": a.template, "values": changes}),
                        "точная модель")
    files, result = dict(job.files), job.result
    brief = latest(ctx, "brief")
    data = {"method": "parametric", "template": a.template, "values": result["values"], "quality": result["quality"],
            "faces": result["faces"]}
    asset = ctx.store.add_asset(ctx.chat_id, "mesh", f"Чертёж: {result['title']}", files, data,
                                parent=brief["id"] if brief else None)
    changed = ", ".join(f"{k} = {v:g}" for k, v in changes.items()) or "стандартные размеры"
    preview_img = Image.open(ctx.settings.outputs / files["preview"])
    content = [{"type": "text", "text": f"Модель по чертежу «{result['title']}» готова: файл {files['mesh']}, "
                                        f"{result['faces']} треугольников ({changed}). Превью: спереди, 3/4, сбоку.\n"
                                        + parametric.describe(a.template)},
               {"type": "image", "file": llm_images.add(ctx.image_dir, preview_img)}]
    return content, {"kind": "mesh", "asset": asset["id"], "files": files, "quality": result["quality"],
                     "method": "parametric", "template": a.template, "values": result["values"], "warnings": []}


HANDLERS = {"look_up": look_up, "write_brief": write_brief, "generate_images": generate_images, "make_3d": make_3d,
            "build_exact": build_exact, "send_to_blender": send_to_blender}


async def execute(ctx: ToolContext, use: dict) -> dict:
    """Run one tool call; any failure becomes an error result the model can react to. A tool that needs the
    user returns its ``fallback`` result plus ``ask`` (the questions): the turn asks them when it can."""

    def failed(text: str) -> dict:
        return {"type": "tool_result", "tool_use_id": use["id"], "is_error": True,
                "content": [{"type": "text", "text": text}], "ui": {"kind": "error", "text": text}}

    tool, handler = BY_NAME.get(use["name"]), HANDLERS.get(use["name"])
    if tool is None or handler is None:
        return failed(f"Нет инструмента {use['name']}")
    raw = use.get("input")
    if isinstance(raw, dict) and "__invalid_json__" in raw:
        return failed(json.dumps({"INVALID_JSON": raw["__invalid_json__"]}, ensure_ascii=False))
    try:
        args = tool.parse(raw)
    except ValueError as e:
        return failed(f"Неверные параметры {use['name']}: {e}")
    await ctx.publish({"type": "tool_running", "id": use["id"], "name": use["name"], "title": TITLES.get(use["name"], ""),
                       "input": args.model_dump()})
    try:
        content, ui = await handler(ctx, args, use["id"])
    except asyncio.CancelledError:
        raise
    except NeedsAnswer as e:
        return {"type": "tool_result", "tool_use_id": use["id"], "is_error": False,
                "content": [{"type": "text", "text": e.fallback}], "ui": e.ui,
                "ask": {"questions": e.questions, "intro": e.intro, "reference": e.reference, "notice": e.notice,
                        "ui": e.ui}}
    except ToolFailure as e:
        return failed(str(e))
    except UserError as e:
        return failed(str(e))
    except Exception as e:  # noqa: BLE001 - the model gets a readable reason, the log the traceback
        log.exception("tool %s failed", use["name"])
        return failed(friendly(e))
    return {"type": "tool_result", "tool_use_id": use["id"], "is_error": False, "content": content, "ui": ui}
