"""What a named thing looks like: a short description and a picture from Wikipedia (Russian and English).

The local model does not know most characters, brands and landmarks by name («Молния Маквин»), and
the image model cannot draw what nobody described to it. ``find`` looks for the article, takes its
intro and its main picture; the picture then goes to the prompt engineer, who describes the look
in English (``prompt_engineer.write_brief``).

Only the name that is looked up leaves the computer: to ru/en.wikipedia.org (the search) and to
upload.wikimedia.org (the picture). The Studio setting «Искать в интернете» switches it off.

Search: a fuzzy title search first (it forgives typos: «молния маквин» finds «Молния Маккуин»), then
a full-text search. A result counts only when every word of the name is close to a word of its
title: a search engine always returns something, and the wrong article (the weather phenomenon
«Молния») is worse than asking the user.
"""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from PIL import Image

USER_AGENT = "DroneGenStudio/0.4 (local drone-show figure studio; looks up what a named character looks like) httpx"
TIMEOUT = 10.0
MIN_WORD = 0.75  # every word of the name must be at least this close to a word of the title
EXTRACT_CHARS = 900
IMAGE_SIDE = 640
MAX_IMAGE_BYTES = 8 * 1024 * 1024
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
transport = None  # httpx transport override (tests)


class Offline(Exception):
    """No connection to the internet (or to Wikipedia)."""


class Unavailable(Exception):
    """Wikipedia was reached but gave no result (a rate limit, an error page); the text is for the user."""


@dataclass
class Page:
    lang: str
    title: str
    url: str = ""
    text: str = ""
    image_url: str = ""
    other: str = ""  # the title of the same article in the other language
    score: float = 0.0
    rank: int = 99


@dataclass
class Found:
    name: str  # what was looked up
    title: str  # the article's title, in Russian when there is a Russian article
    url: str
    text: str  # the intro in Russian (in English when there is no Russian article)
    text_en: str  # the English intro for the prompt engineer, "" when there is none
    image: Image.Image | None = None


def words(text: str) -> list[str]:
    """Lower-case words without the clarification in brackets: «Молния (завод)» -> ["молния"]."""
    text = re.sub(r"\([^)]*\)", " ", text.lower().replace("ё", "е"))
    found = [w for w in re.split(r"[\W_]+", text) if w]
    return [w for w in found if len(w) > 1] or found


def _stem(word: str) -> str:
    """A Russian word without its ending: «молнию» and «молния» -> «молн», «маквина» -> «маквин»."""
    if len(word) >= 5 and CYRILLIC.search(word):
        for _ in range(2):
            if word[-1] in "аяыиеоуюьй":
                word = word[:-1]
    return word


def _close(a: str, b: str) -> float:
    return SequenceMatcher(None, _stem(a), _stem(b)).ratio()


def score(name: str, title: str) -> float:
    """0..1: how well ``title`` names ``name``; 0 unless every word of the name is close to a word of the title.
    Extra words in the title («Губка Боб Квадратные Штаны») lower the score a little, so the closest title wins."""
    q, t = words(name), words(title)
    if not q or not t:
        return 0.0
    forward = [max(_close(w, x) for x in t) for w in q]
    if min(forward) < MIN_WORD:
        return 0.0
    backward = [max(_close(x, w) for w in q) for x in t]
    return 0.7 * sum(forward) / len(forward) + 0.3 * sum(backward) / len(backward)


def same(a: str, b: str) -> bool:
    """The same name written a little differently («Молния Маквин» and «молнию маквина»)."""
    return bool(a and b) and min(score(a, b), score(b, a)) > 0


PROPS = {"prop": "extracts|pageimages|info|langlinks|pageprops", "exintro": 1, "explaintext": 1, "exlimit": "max",
         "exchars": EXTRACT_CHARS, "piprop": "thumbnail", "pithumbsize": IMAGE_SIDE, "pilicense": "any",
         "inprop": "url", "lllimit": "max", "ppprop": "disambiguation", "redirects": 1}


async def _query(client, lang: str, params: dict) -> dict:
    import httpx

    try:
        r = await client.get(f"https://{lang}.wikipedia.org/w/api.php",
                             params={"action": "query", "format": "json", "formatversion": 2, **params})
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError) as e:
        raise Offline(str(e)) from None
    except httpx.TimeoutException:
        raise Unavailable("Википедия не ответила вовремя") from None
    except httpx.HTTPError as e:
        raise Unavailable(f"сбой связи с Википедией ({type(e).__name__})") from None
    if r.status_code == 429:
        raise Unavailable("Википедия временно ограничила запросы, попробуйте через минуту")
    if r.status_code != 200:
        raise Unavailable(f"Википедия ответила ошибкой {r.status_code}")
    try:
        data = r.json()
    except ValueError:
        raise Unavailable("Википедия прислала непонятный ответ") from None
    if "error" in data:
        raise Unavailable(f"Википедия: {data['error'].get('info', 'ошибка')}")
    return data


def _pages(data: dict, lang: str, name: str) -> list[Page]:
    query = data.get("query") or {}
    aliases: dict[str, list[str]] = {}  # an article found through a redirect also counts by the redirect's title
    for r in query.get("redirects") or []:
        aliases.setdefault(r["to"], []).append(r["from"])
    out = []
    for p in query.get("pages") or []:
        if p.get("missing") or p.get("invalid") or p.get("ns", 0) != 0 or "disambiguation" in (p.get("pageprops") or {}):
            continue
        links = p.get("langlinks") or []
        out.append(Page(lang, p["title"], p.get("fullurl", ""), (p.get("extract") or "").strip(),
                        (p.get("thumbnail") or {}).get("source", ""), links[0]["title"] if links else "",
                        max(score(name, t) for t in [p["title"], *aliases.get(p["title"], [])]), p.get("index", 99)))
    return out


def _best(pages: list[Page]) -> Page | None:
    good = sorted((p for p in pages if p.score > 0), key=lambda p: (-p.score, p.rank))
    return good[0] if good else None


async def _search(client, lang: str, name: str) -> Page | None:
    common = {**PROPS, "lllang": "en" if lang == "ru" else "ru"}
    data = await _query(client, lang, {"generator": "prefixsearch", "gpssearch": name, "gpslimit": 5,
                                       "gpsprofile": "fuzzy", **common})
    best = _best(_pages(data, lang, name))
    if best is None:
        data = await _query(client, lang, {"generator": "search", "gsrsearch": name, "gsrlimit": 5, **common})
        best = _best(_pages(data, lang, name))
    return best


async def _article(client, lang: str, title: str) -> Page | None:
    """The same article in the other language (by its interlanguage link); None when it cannot be read."""
    try:
        data = await _query(client, lang, {**PROPS, "titles": title, "lllang": "en" if lang == "ru" else "ru"})
    except (Offline, Unavailable):
        return None
    pages = _pages(data, lang, title)
    return pages[0] if pages else None


async def _picture(client, url: str) -> Image.Image | None:
    """The article's picture on white (transparent PNGs), or None: an article without a picture is still useful."""
    try:
        r = await client.get(url)
        if r.status_code != 200 or len(r.content) > MAX_IMAGE_BYTES:
            return None
        img = Image.open(io.BytesIO(r.content))
        img.load()
    except Exception:  # noqa: BLE001 - no picture is not a failure of the look-up
        return None
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.getchannel("A"))
        return flat
    return img.convert("RGB")


async def find(name: str, english: str = "") -> Found | None:
    """The article about ``name`` (``english``: its English name, when the model knows it).

    None when there is no such article; Offline without a connection, Unavailable when Wikipedia
    cannot answer now."""
    import httpx

    name, english = name.strip(), english.strip()
    searches = [("ru", name)]
    en_name = english or (name if LATIN.search(name) and not CYRILLIC.search(name) else "")
    if en_name:
        searches.append(("en", en_name))
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT, connect=5.0), headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True, transport=transport) as client:
        results = await asyncio.gather(*(_search(client, lang, q) for lang, q in searches), return_exceptions=True)
        pages = {lang: r for (lang, _), r in zip(searches, results) if isinstance(r, Page)}
        errors = [r for r in results if isinstance(r, Exception)]
        if not pages:
            for e in errors:
                if not isinstance(e, (Offline, Unavailable)):
                    raise e
            if errors:  # Offline only when nothing could be reached at all
                raise next((e for e in errors if isinstance(e, Unavailable)), errors[0])
            return None
        ru, en = pages.get("ru"), pages.get("en")
        if ru and en and ru.other != en.title:  # two different articles: the one closer to its name wins
            ru, en = (None, en) if en.score > ru.score else (ru, None)
        if ru and not en and ru.other:  # the English text of the same article, for the prompt engineer
            en = await _article(client, "en", ru.other)
        if en and not ru and en.other:  # a Russian text for the user, when there is one
            ru = await _article(client, "ru", en.other)
        main = ru or en
        image_url = main.image_url or (en.image_url if en else "")
        image = await _picture(client, image_url) if image_url else None
    return Found(name, main.title, main.url, main.text, en.text if en else "", image)
