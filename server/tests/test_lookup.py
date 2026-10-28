"""Looking up how a named character looks (Wikipedia), with the network replaced by canned answers."""

import asyncio

import httpx
import pytest
from conftest import png_bytes
from PIL import Image

from dronegen_server.agents import lookup

RU_PAGES = [
    {"title": "Молния (машиностроительный завод)", "ns": 0, "index": 2, "extract": "Завод в Москве.",
     "fullurl": "https://ru.wikipedia.org/wiki/Molniya_zavod"},
    {"title": "Молния Маккуин", "ns": 0, "index": 1, "extract": "Молния Маккуин — гоночный автомобиль из «Тачек».",
     "fullurl": "https://ru.wikipedia.org/wiki/McQueen", "thumbnail": {"source": "https://upload.wikimedia.org/mcqueen.png"},
     "langlinks": [{"lang": "en", "title": "Lightning McQueen"}]},
    {"title": "Молния (значения)", "ns": 0, "index": 3, "pageprops": {"disambiguation": ""}},
]
EN_PAGE = {"title": "Lightning McQueen", "ns": 0, "index": 1, "extract": "Lightning McQueen is a red race car.",
           "fullurl": "https://en.wikipedia.org/wiki/Lightning_McQueen", "langlinks": [{"lang": "ru", "title": "Молния Маккуин"}]}


def picture_bytes() -> bytes:
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    img.paste((200, 20, 20, 255), (8, 8, 56, 24))
    return png_bytes(img)


class Wiki:
    """A fake Wikipedia: answers the searches with ``ru``/``en`` pages and serves the picture."""

    def __init__(self, ru=RU_PAGES, en=(EN_PAGE,)):
        self.ru, self.en = list(ru), list(en)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "upload.wikimedia.org":
            return httpx.Response(200, content=picture_bytes(), headers={"content-type": "image/png"})
        assert request.headers["user-agent"].startswith("DroneGenStudio/")
        params = request.url.params
        pages = self.ru if request.url.host.startswith("ru.") else self.en
        if "titles" in params:
            pages = [p for p in pages if p["title"] == params["titles"]]
        return httpx.Response(200, json={"query": {"pages": pages}})


@pytest.fixture()
def wiki(monkeypatch):
    fake = Wiki()
    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(fake))
    return fake


def test_names_match_titles_with_typos_but_not_other_things():
    assert lookup.score("Молния Маквин", "Молния Маккуин") > 0.8  # the user's spelling
    assert lookup.score("молнию маквина", "Молния Маккуин") > 0.8  # another word form
    assert lookup.score("Молния Маквин", "Молния (машиностроительный завод)") == 0  # not every word is there
    assert lookup.score("Молния Маквин", "Чёрная Молния (телесериал)") == 0
    assert lookup.score("человек паук", "Человек-паук") == 1
    assert lookup.score("Губка Боб", "Губка Боб") > lookup.score("Губка Боб", "Губка Боб Квадратные Штаны") > 0
    assert lookup.same("Молния Маквин", "молнию маквина") and not lookup.same("Молния Маквин", "Губка Боб")


def test_find_takes_the_matching_article_its_english_text_and_picture(wiki):
    found = asyncio.run(lookup.find("Молния Маквин", "Lightning McQueen"))
    assert found.title == "Молния Маккуин" and "гоночный" in found.text
    assert found.text_en == "Lightning McQueen is a red race car." and found.url.endswith("/McQueen")
    assert found.image.mode == "RGB" and found.image.getpixel((0, 0)) == (255, 255, 255)  # transparency -> white
    hosts = [r.url.host for r in wiki.requests]
    assert hosts.count("upload.wikimedia.org") == 1
    first = wiki.requests[0].url.params
    assert first["generator"] == "prefixsearch" and first["gpsprofile"] == "fuzzy" and first["pilicense"] == "any"


def test_find_without_english_name_reads_the_english_article_by_its_link(wiki):
    found = asyncio.run(lookup.find("молния маквин"))
    assert found.title == "Молния Маккуин" and found.text_en.startswith("Lightning McQueen")
    assert [r.url.params.get("titles") for r in wiki.requests if r.url.host == "en.wikipedia.org"] == ["Lightning McQueen"]


def test_find_falls_back_to_full_text_search_and_gives_up_on_unrelated_titles(monkeypatch):
    fake = Wiki(ru=[RU_PAGES[0]], en=[])
    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(fake))
    assert asyncio.run(lookup.find("Молния Маквин")) is None  # «Молния (завод)» is not the character
    assert [r.url.params["generator"] for r in fake.requests] == ["prefixsearch", "search"]


def test_find_reports_no_internet_and_wikipedia_limits(monkeypatch):
    def offline(request):
        raise httpx.ConnectError("getaddrinfo failed", request=request)

    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(offline))
    with pytest.raises(lookup.Offline):
        asyncio.run(lookup.find("Молния Маквин", "Lightning McQueen"))
    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(lambda r: httpx.Response(429, text="slow down")))
    with pytest.raises(lookup.Unavailable, match="ограничила"):
        asyncio.run(lookup.find("Молния Маквин"))
