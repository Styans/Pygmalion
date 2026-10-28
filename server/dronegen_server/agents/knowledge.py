"""Reference sheets on traditional Kazakh objects for the prompt engineer (``knowledge/*.md``).

Image models often draw the wrong thing (a Chinese pipa for a dombra, a violin for a kobyz).
Each sheet says what the object looks like, what must be right and the usual mistakes; the
«must be right» list also becomes the art director's checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FOLDER = Path(__file__).with_name("knowledge")


@dataclass(frozen=True)
class Entry:
    key: str
    names: tuple[str, ...]
    text: str
    style: str = ""  # the picture style that gives the best 3D for this object: sheet | render | flat
    template: str = ""  # an exact drawing of this object exists (dronegen_server.parametric)


FRONT = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)


@lru_cache(maxsize=1)
def entries() -> tuple[Entry, ...]:
    out = []
    for path in sorted(FOLDER.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = FRONT.match(raw)
        meta = {}
        if m:
            for line in m.group(1).splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        names = tuple(n.strip().lower() for n in meta.get("names", "").split(",") if n.strip())
        out.append(Entry(path.stem, names, raw[m.end():].strip() if m else raw.strip(), meta.get("style", ""),
                         meta.get("template", "")))
    return tuple(out)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е"))


def match(text: str) -> list[Entry]:
    """Sheets whose names occur in ``text`` (any word form starting with the name's stem)."""
    t = _norm(text)
    found = []
    for e in entries():
        for name in e.names:
            n = _norm(name)
            stem = n[:-1] if len(n) >= 4 and n[-1] in "аяыиеоуь" else n  # домбра -> домбр(у/ой/ы)
            if re.search(r"(?<![\w-])" + re.escape(stem), t):
                found.append(e)
                break
    return found


def as_prompt(chosen: list[Entry] | None = None) -> str:
    chosen = list(entries()) if chosen is None else chosen
    return "\n\n".join(f'<sheet name="{e.key}">\n{e.text}\n</sheet>' for e in chosen)
