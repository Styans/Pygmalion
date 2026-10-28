"""The prompt engineer: turns a request in Russian into a precise English brief for Z-Image.

A separate model call with its own instructions and the reference sheets, answering with a
structured ``ImageBrief``: the subject description, the picture style that suits the 3D method,
and the checks the art director (the chat assistant) applies to every picture.

A named character or thing («Молния Маквин») comes with its reference (``tools.look_up``: the
Wikipedia intro and picture, or the user's own description and picture): the prompt engineer sees
the picture and writes its look into the subject, so the image model draws it without knowing the name.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .. import styles
from ..llm import Provider
from . import knowledge

SYSTEM = """\
You write image prompts for Z-Image-Turbo, a text-to-image model that follows long, literal English \
descriptions. The picture is then turned into a 3D model for a drone light show: hundreds of drones \
draw the figure in the night sky, seen from far away, so the figure needs a bold, clear silhouette, \
a few large recognisable parts and no hair-thin details.

From the request (in Russian) write a brief:
- subject: an exact English description of the object only: overall shape and proportions, every \
important part with exact counts ("exactly two strings and two tuning pegs"), materials and colours. \
No style, lighting, background, camera or quality words - they are added automatically for the \
chosen style. One object, no text or logos, no people unless the request asks for them.
- view: the view that shows the shape best: "front view" for most symmetric figures, "side view" for \
animals in motion, "top view" for flat emblems.
- style: "sheet" - front, left, back and right views on one canvas; the most accurate 3D, the default for \
objects, instruments, buildings and characters. "render" - one shaded picture: for animals in dynamic \
poses and when the request asks for a quick result. "flat" - a flat silhouette with flat colours: for \
logos, emblems, ornaments, letters and other flat signs.
- method: "hunyuan_mv" with style sheet, "hunyuan" with style render, "silhouette" with style flat.
- symmetry: true when the object is left-right symmetric in its view (an instrument, a front-facing \
eagle with spread wings, a building); false for animals in side view and asymmetric poses.
- checks: 2-6 short checks in Russian that a correct picture must pass, about counts and key parts \
("ровно 2 струны", "2 колка на головке", "каплевидный корпус"). Take them from the reference sheet \
when there is one.
- counts: the parts whose number a program will count on the picture: 0-4 large parts that are easy \
to see at the picture's scale and that there are two or more of (tuning pegs, legs, wings, wheels, towers, horns, \
humps; never a single part like one beak or one head), each as a \
Russian noun phrase ("колки на головке грифа"), the number VISIBLE in the picture's view (for a sheet: in the \
front view) - a car seen from the front or from the side shows 2 wheels, not 4; a horse from the side shows 4 legs - \
and where: "top" for details in the \
upper part of an upright object (pegs on a head, horns, crown spikes), "whole" for the rest (legs, wings, \
wheels). Never thin or tiny things (strings, frets, feathers): put those into checks only. An empty list \
is fine.
- negative: comma-separated English words for what must not appear (used by SDXL only).
- title_ru: a 1-3 word Russian name of the object only ("Домбра"), without views, poses or styles. notes_ru: one or two short Russian sentences for \
the user about the choices you made (style, view), without technical words.

When the request is about a traditional Kazakh object below, follow its reference sheet: it lists \
what must be right and the usual mistakes of image models; describe the correct details explicitly \
so the image model cannot fall back to the common mistake.

When the request comes with a reference about a named character or thing (text from Wikipedia or from \
the user, often with its picture), the figure is exactly that thing: look at the picture and describe its \
look in the subject so that an image model that never heard the name draws it right - body shape and \
proportions, colours, eyes or face if it has one, characteristic markings, numbers and accessories. Keep \
the name in the subject as well. Ignore the story and the facts in the text that do not change the look.

<reference_sheets>
{sheets}
</reference_sheets>
"""


class PartCount(BaseModel):
    part: str = Field(description="часть по-русски: «колки на головке грифа»", json_schema_extra={"maxLength": 80})
    expected: int = Field(description="сколько таких частей ВИДНО на картинке в этом ракурсе (машина спереди — 2 колеса)")
    where: Literal["whole", "top"] = Field(
        "whole", description="top — детали в верхней части стоящего предмета (колки на головке, рога, зубцы короны); "
                             "whole — везде (ноги, крылья, колёса)")


class ImageBrief(BaseModel):
    title_ru: str = Field(description="Короткое название фигуры по-русски, 1-3 слова", json_schema_extra={"maxLength": 60})
    subject: str = Field(description="English description of the object only", json_schema_extra={"maxLength": 1200})
    view: str = Field(description="front view, side view, three-quarter view or top view", json_schema_extra={"maxLength": 80})
    style: Literal["sheet", "render", "flat"]
    method: Literal["hunyuan_mv", "hunyuan", "silhouette"]
    symmetry: bool
    checks: list[str] = Field(description="2-6 проверок картинки по-русски", json_schema_extra={"maxItems": 6, "items": {"type": "string", "maxLength": 160}})
    counts: list[PartCount] = Field(description="0-4 крупные детали, число которых проверит программа",
                                    json_schema_extra={"maxItems": 4})
    negative: str = Field(json_schema_extra={"maxLength": 400})
    notes_ru: str = Field(json_schema_extra={"maxLength": 400})

    def prompt(self) -> str:
        return styles.build(self.subject, self.style, self.view)


class NextFigure(BaseModel):
    named: str = Field(description="предмет, который пользователь в новых сообщениях просит сделать, как он написал "
                                   "(1-3 слова); пусто, если предмет не назван", json_schema_extra={"maxLength": 60})
    another: bool = Field(description="true — назван другой предмет, не последняя фигура")


NEXT_SYSTEM = """\
A chat makes figures for a drone show, one after another. Below are the chat's last figure and the user's \
new messages. Find out whether the user asks for another object.
- named: the object the new messages ask to make, in the user's words: an animal, a thing, a character, a \
building («лошадь», «домбру», «Губка Боб»). Leave it empty when they name no object: a reply («давай», «да»), \
a complaint or a change of the last figure (colours, texture, pose, size, view, detail), a request to redraw, \
to send it or to make it more exact. Colours, texture, wings, legs and other parts are never the object.
- another: true only when named is not empty and is not the last figure.

Examples (last figure «Беркут»):
«мне нужна лошадь» -> named «лошадь», another true
«лошадь вместо орла» -> named «лошадь», another true
«проблемы с текстурой» -> named «», another false
«давай» -> named «», another false
«сделай крылья пошире» -> named «», another false
«сделай лист из 4 видов» -> named «», another false
«нарисуй беркута ещё раз» -> named «беркут», another false"""
# what the user may name that is not a figure: a view sheet, a picture, a model, its texture...
NOT_FIGURES = ("лист", "вид", "картин", "рисун", "модел", "вариант", "текстур", "сетк")


def stems(text: str) -> list[str]:
    """Word stems that match other forms of the word: лошадь -> лоша (лошади), домбра -> домб (домбру)."""
    return [w[:max(3, len(w) - 2)] for w in re.findall(r"\w+", text.lower().replace("ё", "е")) if len(w) >= 3]


def mentions(text: str, words: list[str]) -> bool:
    t = text.lower().replace("ё", "е")
    return any(re.search(r"(?<!\w)" + re.escape(w), t) for w in words)


async def same_figure(provider: Provider, figure: str, said: list[str]) -> bool:
    """Whether what the user typed since the last brief is still about its figure («проблемы с текстурой») or asks
    for another one («мне нужна лошадь» after an eagle: carried into the eagle's request, it drew an eagle again).
    The model gets the figure's name only: from the words of its request («нужен орел») it «named» the eagle."""
    text = f"Last figure: «{figure}».\nThe user wrote since: " + " / ".join(f"«{w}»" for w in said)
    answer = await provider.structured(NEXT_SYSTEM, [{"role": "user", "content": [{"type": "text", "text": text}]}],
                                       NextFigure, "low")
    # another figure counts only when the user typed its name, and it is neither the last figure (беркута) nor a
    # sheet or a texture (a small model took «лист из 4 видов» for an object)
    named = stems(" ".join(w for w in answer.named.lower().split() if not w.startswith(NOT_FIGURES)))
    return not (answer.another and mentions(" ".join(said), named) and not mentions(figure, named))


def system_prompt(request: str, all_sheets: bool) -> str:
    """With Claude every sheet goes in (the prompt is cached); the small local model gets the matching ones."""
    chosen = None if all_sheets else knowledge.match(request)
    return SYSTEM.replace("{sheets}", knowledge.as_prompt(chosen) or "(none)")


CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LANGUAGES = ("Write subject, view and negative in English; title_ru, checks and notes_ru in Russian.")


STYLE_RU = {"sheet": "лист из 4 видов: спереди, слева, сзади, справа", "render": "один объёмный рисунок", "flat": "плоский силуэт"}
METHOD_RU = {"hunyuan_mv": "Hunyuan3D по карточкам видов", "hunyuan": "Hunyuan3D по одной картинке",
             "silhouette": "«Силуэт → объём»"}
METHOD_OF = {"sheet": "hunyuan_mv", "render": "hunyuan", "flat": "silhouette"}

# What the user says that fixes the picture style (and with it the 3D method). The first group
# that matches wins; the newest message is looked at first.
STYLE_WORDS = (
    ("sheet", ("лист видов", "лист", "карточк", "со всех сторон", "с разных сторон", "максимально точн", "самая точн")),
    ("flat", ("силуэт", "плоск", "контур", "эмблем", "логотип")),
    ("render", ("быстро", "быстрый", "быстрая", "одной картинк", "одна картинка", "без листа")),
)


# the «Точность» switch of a chat (state["precision"]); «auto» leaves the choice to the words and the object
PRECISION_STYLE = {"single": "render", "sheet": "sheet"}


def style_hint(texts: list[str], precision: str | None = None) -> str | None:
    """The style the user asked for: with the switch in the chat, else in their own words (newest first), or None."""
    if precision in PRECISION_STYLE:
        return PRECISION_STYLE[precision]
    for text in reversed(texts):
        t = text.lower().replace("ё", "е")
        for style, words in STYLE_WORDS:
            if any(w in t for w in words):
                return style
    return None


SURFACE_WORDS = (
    ("lowpoly", ("low-poly", "lowpoly", "low poly", "лоу-поли", "лоуполи", "лоу поли", "граненн", "полигональн")),
    ("as_is", ("как есть", "без обработки", "без сглаживания", "сырая сетка")),
    ("smooth", ("гладк", "сглаж")),
)


def surface_hint(texts: list[str]) -> str:
    """How to treat the mesh surface, from the user's own words (newest message first).
    Default: smooth - a clean, closed surface of ~70k faces instead of the raw ~250k noisy ones."""
    for text in reversed(texts):
        t = text.lower().replace("ё", "е")
        for surface, words in SURFACE_WORDS:
            if any(w in t for w in words):
                return surface
    return "smooth"


def explain(brief: ImageBrief) -> str:
    """The note for the user, built from the final fields (the small local model's own note
    sometimes contradicted its choices)."""
    return (f"Рисую: {STYLE_RU[brief.style]}. 3D: {METHOD_RU[brief.method]}"
            + (", с симметрией" if brief.symmetry else "") + ".")


SOURCE_EN = {"wikipedia": "from Wikipedia", "user": "from the user"}


def reference_blocks(references) -> list[dict]:
    """The references of the figure as message blocks: their text and their pictures."""
    blocks = []
    for r in references:
        text = r.get("text_en") or r.get("text") or ""
        head = f"Reference about «{r.get('title') or r.get('name')}» ({SOURCE_EN.get(r.get('source'), 'reference')})"
        blocks.append({"type": "text", "text": f"{head}:\n{text}" if text else f"{head}: only the picture below."})
        if r.get("picture"):
            blocks.append({"type": "text", "text": "Its picture (describe the look from it):"})
            blocks.append({"type": "image", "file": r["picture"]})
    return blocks


async def write_brief(provider: Provider, request: str, style: str | None = None, effort: str = "medium",
                      references=()) -> ImageBrief:
    """``style``: what the user asked for (``style_hint``); else the known object's style, else the model's.
    ``references``: the data of the figure's reference assets (``tools.look_up``)."""
    text = f"Request: {request.strip()}"
    if style:
        text += f"\nThe user asked for the picture style: {style}."
    text += "\n\n" + LANGUAGES
    system = system_prompt(request, all_sheets=provider.name == "claude")
    messages = [{"role": "user", "content": [*reference_blocks(references), {"type": "text", "text": text}]}]
    brief = await provider.structured(system, messages, ImageBrief, effort)
    if CYRILLIC.search(brief.subject + brief.view + brief.negative):  # small models answer in the request's language
        retry = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": brief.model_dump_json()}]},
            {"role": "user", "content": [{"type": "text", "text": "subject, view and negative must be in English: "
                                                                  "the image model understands English only. Rewrite."}]},
        ]
        brief = await provider.structured(system, retry, ImageBrief, effort)
    known = [e for e in knowledge.match(request) if e.style]
    if style in METHOD_OF:  # the user's words win
        brief.style = style
    elif len(known) == 1:  # the sheet of a known object says which style suits it best
        brief.style = known[0].style
    brief.method = METHOD_OF[brief.style]
    if brief.style == "sheet":
        brief.view = "front, left, back and right views"  # the sheet shows all of them
    brief.notes_ru = explain(brief)
    return brief
