"""The art director: does a generated picture have the right number of parts?

Asked "is it right: 2 pegs?", the small local model answers yes to everything - it repeats the
expected answer (it approved dombra pictures with four pegs). Asked for the number in a strict
JSON field it guesses (counted 1-3 on pictures with 4 pegs). What works (calibrated on pictures
with known truth, docs/BENCHMARKS.md):

* a neutral question that does not contain the expected number;
* the model describing every part one by one in plain text, the total on the last line;
* for details on the top of an upright object (pegs, horns) a close crop of that band: counting on
  the whole card was right 5 times of 8, on the crop 7 of 8;
* several independent counts (crop, crop asked per side, the whole picture) and a picture passes
  only when ALL of them equal the expected number: 0 of 8 wrong pictures passed.

The cost is some false alarms (a good picture rejected), which only means another candidate.
Counting large parts (pegs, legs, wings, wheels, towers) works; thin things (strings) are only
listed in the brief's text checks.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from PIL import Image
from pydantic import BaseModel

from ..llm import LLMError, Provider
from ..llm import images as llm_images

SYSTEM = "Ты внимательно рассматриваешь картинки и считаешь детали. Отвечай по-русски, коротко."
QUESTION = ("Посчитай на картинке: {part}.\n"
            "Опиши каждую деталь по очереди (где она находится: слева или справа, выше или ниже). "
            "В конце напиши отдельную строку «Итого: N», где N — сколько всего деталей ты насчитал.")
QUESTION_SIDES = ("Посчитай на картинке: {part}.\n"
                  "Сначала опиши, что видно слева от центральной оси (каждую деталь отдельно), потом — что справа. "
                  "В конце напиши отдельную строку «Итого: N», где N — сколько всего деталей на обеих сторонах.")
QUESTION_ORDER = ("Сколько на картинке: {part}?\n"
                  "Назови каждую по порядку («первая…, вторая…») и отметь, где она. "
                  "В конце напиши отдельную строку «Итого: N».")
TOTAL = re.compile(r"Итого[^\d\n]{0,12}(\d+)")
TOP_BAND = 0.28  # the top part of the figure that holds pegs, horns, a crown
WHERES = ("whole", "top")


class CheckItem(BaseModel):
    check: str
    passed: bool
    note: str


class Verdict(BaseModel):
    seen: str
    checks: list[CheckItem]
    ok: bool


def parse_total(text: str) -> int | None:
    found = TOTAL.findall(text)
    return int(found[-1]) if found else None


def short(text: str, limit: int = 140) -> str:
    one = re.sub(r"\s+", " ", TOTAL.sub("", text)).strip(" .,;:-—")
    return one if len(one) <= limit else one[: limit - 1].rstrip() + "…"


def figure_bbox(img: Image.Image, tolerance: int = 14) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of what differs from the border colour (the background)."""
    rgb = np.asarray(img.convert("RGB"), np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    mask = np.abs(rgb - bg).max(axis=-1) > tolerance
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return 0, 0, img.width, img.height
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def band(img: Image.Image, where: str, size: int = 768) -> Image.Image:
    """A close view of the part of the figure where ``where`` says the detail is."""
    if where != "top":
        return img
    l, t, r, b = figure_bbox(img)
    pad = max(4, int(0.04 * (r - l)))
    crop = img.convert("RGB").crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, t + int(TOP_BAND * (b - t)) + pad)))
    k = size / max(crop.size)
    return crop.resize((max(1, round(crop.width * k)), max(1, round(crop.height * k))), Image.LANCZOS)


def variants(where: str) -> list[tuple[str, str]]:
    """(which picture, which question) for the independent counts of one part."""
    if where == "top":
        return [("band", QUESTION), ("band", QUESTION_SIDES), ("whole", QUESTION)]
    return [("whole", QUESTION), ("whole", QUESTION_ORDER)]


async def check(provider: Provider, image_dir, image_file: str, parts: list[tuple[str, int, str]]) -> Verdict | None:
    """``parts``: (name of the part, expected number, where: "whole" | "top"). The verdict for one
    picture of the chat's image store, or None when there is nothing to count."""
    if not parts:
        return None
    img = Image.open(llm_images.path(image_dir, image_file))
    img.load()
    seen, checks = [], []
    for name, expected, where in parts:
        pictures = {"whole": image_file}
        if where == "top":
            pictures["band"] = llm_images.add(image_dir, band(img, where))
        counts, answers = [], []
        for picture, question in variants(where):
            messages = [{"role": "user", "content": [{"type": "image", "file": pictures[picture]},
                                                     {"type": "text", "text": question.format(part=name)}]}]
            try:
                answer = await provider.text(SYSTEM, messages, "low")
            except LLMError:
                answer = ""
            answers.append(answer)
            counts.append(parse_total(answer))
        label = f"{name}: должно быть {expected}"
        known = [c for c in counts if c is not None]
        if not known:
            checks.append(CheckItem(check=label, passed=False, note="не удалось посчитать"))
            continue
        if len(set(known)) == 1:
            got, note = known[0], f"видно {known[0]} ({short(answers[counts.index(known[0])])})"
        else:  # the counts disagree: not sure -> not passed
            got = Counter(known).most_common(1)[0][0]
            note = f"ответы расходятся: {', '.join(map(str, known))}"
        seen.append(f"{name} — {got}")
        passed = len(known) == len(counts) and set(known) == {expected}
        checks.append(CheckItem(check=label, passed=passed, note=note))
    if not seen:
        return None
    return Verdict(seen="; ".join(seen), checks=checks, ok=all(c.passed for c in checks))


def summary(verdict: Verdict | None) -> str:
    if verdict is None:
        return "проверка не проводилась"
    if verdict.ok:
        return f"количество деталей верное. Посчитано: {verdict.seen}"
    failed = [f"{c.check} ({c.note})" for c in verdict.checks if not c.passed]
    return "НЕ прошла, неверное количество: " + "; ".join(failed)
