"""The Studio chat end to end with a scripted model: questions, brief, pictures, 3D, Blender inbox."""

import asyncio
import itertools
import json
import time

import pytest
from conftest import png_bytes, ring_image

from dronegen_server import imaging
from dronegen_server.agents import knowledge, prompt_engineer
from dronegen_server.agents.prompt_engineer import ImageBrief, NextFigure, PartCount
from dronegen_server.llm import Event, LLMError, Provider
from dronegen_server.store import Store, StudioConfig

BRIEF = ImageBrief(title_ru="Машинка", subject="a cartoon race car with big round wheels", view="side view",
                   style="flat", method="silhouette", symmetry=False, checks=["4 колеса", "одна машина"],
                   counts=[PartCount(part="колёса", expected=4)], negative="text", notes_ru="Сделаю плоский силуэт.")


class Scripted(Provider):
    """A model that answers from a script; a step is a list of blocks, a callable or an exception."""

    def __init__(self, script, name="claude"):
        self.script, self.name, self.title = list(script), name, f"fake-{name}"
        self.seen: list[list[dict]] = []
        self.briefs: list[list[dict]] = []  # what the prompt engineer was given

    async def respond(self, system, messages, tools, effort, on_event):
        self.seen.append(messages)
        step = self.script.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, float):  # think for a while (to test stopping)
            await asyncio.sleep(step)
            step = [{"type": "text", "text": "поздно"}]
        for b in step:
            if b["type"] == "text":
                await on_event(Event("text", b["text"]))
            elif b["type"] == "tool_use":
                await on_event(Event("tool_start", data={"id": b["id"], "name": b["name"]}))
        stop = "tool_use" if any(b["type"] == "tool_use" for b in step) else "end_turn"
        return {"role": "assistant", "content": step, "provider": self.name, "model": "fake", "stop": stop, "usage": {}}

    async def structured(self, system, messages, model, effort):
        if model is NextFigure:  # a horse is another figure; whatever else the user writes changes the last one
            said = messages[0]["content"][0]["text"].split("The user wrote since:")[1]
            return NextFigure(named="лошадь", another=True) if "лошад" in said else NextFigure(named="", another=False)
        assert model is ImageBrief
        self.briefs.append(messages)
        return BRIEF.model_copy(deep=True)  # write_brief changes the brief it gets: not the shared one

    async def text(self, system, messages, effort="low"):
        assert messages[0]["content"][0]["type"] == "image"  # the art director looks
        assert "4" not in messages[0]["content"][1]["text"]  # ... and is not told the expected number
        return "Первое колесо слева, второе справа, третье, четвёртое.\nИтого: 4"


CALLS = itertools.count()


def use(tool, **inp):
    return {"type": "tool_use", "id": f"call_{tool}_{next(CALLS)}", "name": tool, "input": inp}


def last_result(messages, name):
    """The newest tool_result content for the newest call of ``name`` in the history the model saw."""
    ids = [b["id"] for m in messages if m["role"] == "assistant" for b in m["content"]
           if b["type"] == "tool_use" and b["name"] == name]
    for m in reversed(messages):
        for b in m["content"]:
            if b.get("type") == "tool_result" and b["tool_use_id"] == ids[-1]:
                return b
    raise AssertionError(f"no result for {name}")


@pytest.fixture()
def studio_client(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from dronegen_server.app import create_app

    monkeypatch.setattr(imaging, "rembg_available", lambda: False)  # never download u2net in tests
    app = create_app(settings)
    with TestClient(app) as c:
        app.state.studio.config.update(image_backend="mock")
        yield c, app.state.studio


def wait_turn(studio, chat_id, timeout=60):
    t0 = time.time()
    while studio.chat.busy(chat_id):
        assert time.time() - t0 < timeout, "turn did not finish"
        time.sleep(0.05)


def history_is_valid(messages):
    """Every tool call is answered by a tool result in the next user message (notices are for people only:
    the providers leave them out)."""
    messages = [m for m in messages if m["role"] != "notice"]
    for i, m in enumerate(messages):
        calls = [b["id"] for b in m["content"] if b["type"] == "tool_use"] if m["role"] == "assistant" else []
        if calls and i + 1 < len(messages):
            results = {b["tool_use_id"] for b in messages[i + 1]["content"] if b["type"] == "tool_result"}
            assert set(calls) == results, (calls, results)


def test_store_and_config(tmp_path):
    s = Store(tmp_path)
    p = s.add_project("Наурыз-27")
    c = s.add_chat(p["id"])
    s.add_message(c["id"], {"role": "user", "content": [{"type": "text", "text": "домбра"}]})
    assert s.messages(c["id"])[0]["content"][0]["text"] == "домбра"
    assert [x["id"] for x in s.chats(query="ДОМБР")] == [c["id"]]  # finds message text, any case
    a = s.add_asset(c["id"], "image", "Вариант 1", {"image": "x.png"})
    assert s.asset_by_file(c["id"], "x.png")["id"] == a["id"]
    item = s.add_inbox("m.glb", "Домбра", c["id"], a["id"])
    assert s.inbox()[0]["id"] == item["id"]
    s.update_inbox(item["id"], "imported", "DG Домбра")
    assert s.inbox() == [] and s.inbox(None)[0]["object"] == "DG Домбра"
    s.set_state(c["id"], pending={"x": 1})
    assert s.chat(c["id"])["state"] == {"pending": {"x": 1}}
    s.delete_project(p["id"])
    assert s.chats() == []
    cfg = StudioConfig(tmp_path)
    cfg.set_secret("anthropic_api_key", "sk-ant-secret-1234")
    assert cfg.secrets_state()["anthropic_api_key"] == {"set": True, "hint": "…1234"}
    assert cfg.update(provider="local", nonsense=1)["provider"] == "local" and "nonsense" not in cfg.all()


def test_knowledge_matches_word_forms():
    assert [e.key for e in knowledge.match("Нарисуй домбру с орнаментом")] == ["dombra"]
    assert {e.key for e in knowledge.match("тулпар над юртой")} == {"tulpar", "yurt"}
    assert knowledge.match("машинка") == []
    assert "<sheet" in prompt_engineer.system_prompt("x", all_sheets=True)
    assert "dombra" in prompt_engineer.system_prompt("домбра", all_sheets=False)
    assert "kobyz" not in prompt_engineer.system_prompt("домбра", all_sheets=False)


def test_full_dialog(studio_client):
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={"title": "Шоу"}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    state = {}

    def pick_variant(messages):
        res = last_result(messages, "generate_images")
        assert not res["is_error"], res
        files = [c["text"].split("файл ")[1].split(",")[0] for c in res["content"] if c["type"] == "text" and "Вариант" in c["text"]]
        assert len(files) == 2 and sum(c["type"] == "image" for c in res["content"]) == 2  # the model sees both pictures
        verdicts = [c["text"] for c in res["content"] if c["type"] == "text" and c["text"].startswith("Проверка")]
        assert len(verdicts) == 2 and "количество деталей верное" in verdicts[0]
        assert res["ui"]["items"][0]["check"]["ok"] is True
        state["image"] = files[0]
        return [{"type": "text", "text": "Беру первый вариант."}, use("make_3d", image=files[0])]

    def send(messages):
        res = last_result(messages, "make_3d")
        assert not res["is_error"], res
        assert any(c["type"] == "image" for c in res["content"])  # the preview of the model
        state["mesh"] = res["ui"]["files"]["mesh"]
        return [use("send_to_blender", mesh=state["mesh"], name="Машинка")]

    fake = Scripted([
        [{"type": "text", "text": "Уточню пару моментов."},
         use("ask_user", questions=[{"question": "Какая машина?", "options": ["Гоночная", "Грузовик"]}])],
        lambda m: [use("write_brief", request="мультяшная гоночная машина: " + json.dumps(m[-1]["content"][0]["content"],
                                                                                         ensure_ascii=False))],
        [use("generate_images", subject=BRIEF.subject, count=2)],
        pick_variant,
        send,
        [{"type": "text", "text": "Готово: модель в очереди для Blender."}],
    ])
    studio.providers.pick = lambda: fake

    r = client.post(f"/studio/api/chats/{cid}/messages", data={"text": "сделай машины в мультяшном стиле"})
    assert r.status_code == 202
    wait_turn(studio, cid)
    chat = client.get(f"/studio/api/chats/{cid}").json()
    assert chat["pending"]["questions"][0]["options"] == ["Гоночная", "Грузовик"]
    assert client.post(f"/studio/api/chats/{cid}/messages", data={"text": "a"}).status_code == 202  # answering is allowed
    wait_turn(studio, cid)
    # the free text became the answer to the question
    answer = fake.seen[1][-1]["content"][0]
    assert answer["type"] == "tool_result" and "a" in answer["content"][0]["text"]

    chat = client.get(f"/studio/api/chats/{cid}").json()
    assert chat["chat"]["title"] == "Машинка"  # named after the brief
    kinds = [a["kind"] for a in chat["assets"]]
    assert kinds.count("brief") == 1 and kinds.count("image") == 2 and kinds.count("mesh") == 1
    mesh = next(a for a in chat["assets"] if a["kind"] == "mesh")
    assert mesh["parent"] == next(a for a in chat["assets"] if a["files"].get("image") == state["image"])["id"]
    history_is_valid(chat["messages"])
    assert all("raw" not in m for m in chat["messages"])
    assert chat["messages"][-1]["content"][0]["text"].startswith("Готово")

    inbox = client.get("/blender/inbox").json()
    assert [i["name"] for i in inbox] == ["Машинка"] and inbox[0]["file"] == state["mesh"]
    assert client.get(f"/files/{state['mesh']}").status_code == 200
    done = client.post(f"/blender/inbox/{inbox[0]['id']}/done", json={"object": "DG Машинка"}).json()
    assert done["status"] == "imported" and client.get("/blender/inbox").json() == []


def test_stop_closes_open_calls(studio_client, monkeypatch):
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    fake = Scripted([[use("write_brief", request="звезда")], [use("generate_images", subject="a star", count=4)], 30.0])
    studio.providers.pick = lambda: fake
    mock = client.app.state.manager.registry.get("txt2img", "mock")
    monkeypatch.setitem(type(mock).defaults, "steps", 150)  # slow pictures: 4.5 s each
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "звезда"})
    t0 = time.time()
    while not any(b.get("type") == "tool_use" for m in studio.store.messages(cid) for b in m["content"]):
        assert time.time() - t0 < 10
        time.sleep(0.05)
    assert client.post(f"/studio/api/chats/{cid}/messages", data={"text": "ещё"}).status_code == 409
    assert client.post(f"/studio/api/chats/{cid}/stop").json() == {"stopped": True}
    messages = studio.store.messages(cid)
    history_is_valid(messages)
    assert messages[-1]["content"][0]["is_error"] and "Остановлено" in messages[-1]["content"][0]["content"][0]["text"]
    jobs = [j for j in client.app.state.manager.recent(10) if j.kind == "txt2img"]
    t0 = time.time()
    while any(j.status in ("queued", "running") for j in jobs) and time.time() - t0 < 5:
        time.sleep(0.05)  # the worker notices the cancel at the next step of the running job
    assert len(jobs) == 4 and all(j.status == "cancelled" for j in jobs)


def test_offline_switch_to_local(studio_client):
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    claude = Scripted([LLMError("Нет связи с Claude API (проверьте интернет)", offline=True)])
    local = Scripted([[{"type": "text", "text": "Привет из офлайна"}]], name="local")
    studio.providers.pick = lambda: claude
    studio.providers.fallback = lambda current: local if current is claude else None
    events = studio.chat.subscribe(cid)
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "привет"})
    wait_turn(studio, cid)
    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    notices = [e["text"] for e in seen if e["type"] == "notice"]
    assert any("локальной модели" in n for n in notices)
    assert [e["name"] for e in seen if e["type"] == "provider"] == ["claude", "local"]
    assert seen[-1] == {**seen[-1], "type": "turn_end", "status": "done"}
    assert studio.store.messages(cid)[-1]["provider"] == "local"


def test_uploaded_photo_and_errors(studio_client):
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    fake = Scripted([lambda m: [use("make_3d", image="nope.png")],
                     lambda m: [{"type": "text", "text": last_result(m, "make_3d")["content"][0]["text"]}]])
    studio.providers.pick = lambda: fake
    files = [("files", ("photo.png", png_bytes(ring_image()), "image/png"))]
    assert client.post(f"/studio/api/chats/{cid}/messages", data={"text": "вот фото"}, files=files).status_code == 202
    wait_turn(studio, cid)
    first = studio.store.messages(cid)[0]["content"]
    assert first[0]["type"] == "image" and first[0]["source"].endswith("_upload.png")
    assert client.get(f"/studio/api/images/{first[0]['file']}").status_code == 200
    assert "не найден" in studio.store.messages(cid)[-1]["content"][0]["text"]  # the tool error reached the model
    assert client.post(f"/studio/api/chats/{cid}/messages", data={"text": " "}).status_code == 400
    assert client.get("/studio/api/chats/nope").status_code == 404


def test_settings_never_return_secrets(studio_client):
    client, studio = studio_client
    r = client.put("/studio/api/settings", json={"anthropic_api_key": "sk-ant-abcdef-9876", "provider": "claude"})
    body = r.json()
    assert "sk-ant" not in json.dumps(body) and body["secrets"]["anthropic_api_key"]["hint"] == "…9876"
    assert body["providers"]["claude"]["available"] and body["providers"]["current"] == "claude"
    assert client.put("/studio/api/settings", json={"provider": "gpt"}).status_code == 400
    client.put("/studio/api/settings", json={"anthropic_api_key": ""})
    assert client.get("/studio/api/status").json()["secrets"]["anthropic_api_key"] == {"set": False}


def test_art_director_counts_without_being_told_the_answer(tmp_path):
    from dronegen_server.agents import art_director
    from dronegen_server.llm import images as llm_images
    from PIL import Image, ImageDraw

    asked = []

    class Counter(Provider):
        def __init__(self, pegs):
            self.pegs = pegs

        async def text(self, system, messages, effort="low"):
            question = messages[0]["content"][1]["text"]
            asked.append((messages[0]["content"][0]["file"], question))
            if "колки" in question:
                n = self.pegs.pop(0) if self.pegs else 4
                return "Слева выше — первый. Справа выше — второй.\nИтого: %d" % n
            return "Две ноги.\nИтого колков: 2"

    img = Image.new("RGB", (300, 600), (200, 200, 200))
    ImageDraw.Draw(img).rectangle([120, 100, 180, 560], fill=(120, 60, 20))
    picture = llm_images.add(tmp_path, img)
    parts = [("колки на головке", 2, "top"), ("ноги", 2, "whole")]
    # all three counts of the pegs agree with the brief -> passed
    ok = asyncio.run(art_director.check(Counter([2, 2, 2]), tmp_path, picture, parts))
    assert ok.ok and ok.checks[0].passed and "видно 2" in ok.checks[0].note
    assert len([q for _, q in asked if "колки" in q]) == 3 and len([q for _, q in asked if "ноги" in q]) == 2
    # the crop (a different picture) is used for two of the three counts, the whole card for the third
    pegs_files = [f for f, q in asked[:5] if "колки" in q]
    assert pegs_files[0] == pegs_files[1] != pegs_files[2] == picture
    assert all("2" not in q for _, q in asked)  # the expected numbers never reach the model
    # one disagreeing count is enough to fail: «not sure» is not a pass
    asked.clear()
    shaky = asyncio.run(art_director.check(Counter([2, 4, 2]), tmp_path, picture, parts))
    assert shaky.ok is False and [c.passed for c in shaky.checks] == [False, True]
    assert "расходятся: 2, 4, 2" in shaky.checks[0].note
    wrong = asyncio.run(art_director.check(Counter([4, 4, 4]), tmp_path, picture, parts[:1]))
    text = art_director.summary(wrong)
    assert text.startswith("НЕ прошла") and "колки на головке: должно быть 2 (видно 4 (" in text
    assert art_director.summary(None) == "проверка не проводилась"
    assert asyncio.run(art_director.check(Counter([]), tmp_path, picture, [])) is None

    class Silent(Provider):  # no answer at all: nothing counted, not a crash
        async def text(self, system, messages, effort="low"):
            raise LLMError("нет связи")

    assert asyncio.run(art_director.check(Silent(), tmp_path, picture, parts[:1])) is None
    assert art_director.parse_total("1, 2, 3.\nИтого: 3") == 3 and art_director.parse_total("Итого колков: 12") == 12
    assert art_director.parse_total("Итого: 2. Нет, Итого: 4") == 4 and art_director.parse_total("не знаю") is None
    assert art_director.short("Слева первый. Справа второй.\nИтого: 2", 20) == "Слева первый. Справ…"


def test_figure_band_is_the_top_of_the_figure():
    from dronegen_server.agents import art_director
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 800), (200, 200, 200))
    ImageDraw.Draw(img).rectangle([150, 100, 250, 700], fill=(10, 10, 10))
    assert art_director.figure_bbox(img) == (150, 100, 251, 701)
    top = art_director.band(img, "top")
    assert top.height == 768 and top.width < 768  # a close view of the top ~28 % of the tall figure
    assert art_director.band(img, "whole") is img


def test_known_objects_get_their_style_and_the_note_matches(monkeypatch):
    """The small model picked «render» and wrote «sheet» in its note: the knowledge sheet decides, the note follows."""
    class Sloppy(Provider):
        name = "local"

        async def structured(self, system, messages, model, effort):
            return BRIEF.model_copy(update={"style": "render", "method": "hunyuan", "notes_ru": "Стиль — листовой."})

    dombra = asyncio.run(prompt_engineer.write_brief(Sloppy(), "Сделай домбру"))
    assert (dombra.style, dombra.method) == ("sheet", "hunyuan_mv")
    assert dombra.notes_ru == "Рисую: лист из 4 видов: спереди, слева, сзади, справа. 3D: Hunyuan3D по карточкам видов."
    assert dombra.prompt().startswith("character model sheet")
    unknown = asyncio.run(prompt_engineer.write_brief(Sloppy(), "Сделай машинку"))
    assert (unknown.style, unknown.method) == ("render", "hunyuan")  # nothing known: the model's choice stays
    chosen = asyncio.run(prompt_engineer.write_brief(Sloppy(), "Сделай домбру", style="flat"))
    assert (chosen.style, chosen.method) == ("flat", "silhouette") and "плоский силуэт" in chosen.notes_ru
    assert {e.key: e.style for e in knowledge.entries()}["shanyrak"] == "flat"
    two = asyncio.run(prompt_engineer.write_brief(Sloppy(), "домбра и кобыз вместе"))
    assert two.style == "render"  # two known objects: no single style, the model's choice stays


def test_another_figure_only_when_the_user_names_it():
    """The local model «named» the eagle (орёл, from the request of «Беркут») after «да» or «не похоже» and called it
    another figure: another figure counts only when the user typed its name, and it is not the last figure."""
    class Says(Provider):
        def __init__(self, named, another):
            self.answer = NextFigure(named=named, another=another)

        async def structured(self, system, messages, model, effort):
            assert model is NextFigure
            return self.answer

    def same(named, another, said):
        return asyncio.run(prompt_engineer.same_figure(Says(named, another), "Беркут", said))

    assert same("орел", True, ["да"]) and same("орел", True, ["не похоже"])
    assert same("беркута", True, ["нарисуй беркута ещё раз"])
    assert same("", False, ["проблемы с текстурой"]) and same("", True, ["давай"])
    assert not same("лошадь", True, ["мне нужна лошадь"]) and not same("лошадь", True, ["мне нужна модель лошади"])
    assert not same("домбру", True, ["цвета плохие", "а теперь домбру"])
    assert same("лошадь", False, ["лошадь"])  # the model's own «same» still counts
    assert same("лист", True, ["сделай лист из 4 видов"]) and same("модель", True, ["модель кривая"])


def test_style_words_of_the_user():
    hint = prompt_engineer.style_hint
    assert hint(["Домбра для финала шоу"]) is None  # nothing about the style: the model never decides it from "шоу"
    assert hint(["сделай плоский силуэт"]) == "flat" and hint(["нужна эмблема"]) == "flat"
    assert hint(["дай лист видов, максимально точно"]) == "sheet" and hint(["с разных сторон"]) == "sheet"
    assert hint(["быстро, по одной картинке"]) == "render"
    assert hint(["Ответы: «Стиль» — Плоский силуэт", "а теперь сделай лист видов"]) == "sheet"  # the newest message wins
    assert hint(["лист видов", "и без всяких глупостей"]) == "sheet"


def test_event_streams_end_when_the_server_stops(studio_client):
    """An open Studio window must not keep the server from shutting down."""
    from types import SimpleNamespace

    client, studio = studio_client
    server = SimpleNamespace(should_exit=False)
    client.app.state.server = server
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    done = []

    def listen(url):
        with client.stream("GET", url) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            lines = [line for line in r.iter_lines() if line]
        done.append((url, lines))

    import threading

    threads = [threading.Thread(target=listen, args=(u,), daemon=True)
               for u in ("/studio/api/events", f"/studio/api/chats/{cid}/events")]
    for t in threads:
        t.start()
    time.sleep(1.5)
    assert not done  # still open
    server.should_exit = True
    for t in threads:
        t.join(timeout=5)
    assert len(done) == 2 and all(lines[0].startswith("event: hello") for _, lines in done)


def test_surface_words_of_the_user():
    hint = prompt_engineer.surface_hint
    assert hint(["Домбра для финала шоу"]) == "smooth"  # the default: clean and closed
    assert hint(["сделай в стиле лоу-поли"]) == "lowpoly" and hint(["Low-poly пожалуйста"]) == "lowpoly"
    assert hint(["без сглаживания, как есть"]) == "as_is"
    assert hint(["сделай гладко"]) == "smooth"
    assert hint(["лоу-поли", "нет, сделай гладкую"]) == "smooth"  # the newest message wins


def test_user_photos_from_several_sides_go_to_hunyuan_mv(studio_client):
    """The user's own photos are cards: the method switches to the multi-view model (not installed here -> a clear error)."""
    from PIL import Image

    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    Image.new("RGB", (64, 64), (200, 40, 40)).save(studio.settings.outputs / "drawn.png")
    files = {}

    def photos(messages):
        files["front"], files["side"] = [b["source"] for m in messages if m["role"] == "user" for b in m["content"]
                                         if b.get("type") == "image"]
        return [use("make_3d", image=files["front"], photos=[{"view": "left", "file": files["side"]}])]

    fake = Scripted([
        photos,
        lambda m: [use("make_3d", image=files["front"], photos=[{"view": "left", "file": files["side"]},
                                                                {"view": "left", "file": files["front"]}])],
        lambda m: [use("make_3d", image=files["front"], photos=[{"view": "top", "file": files["side"]}])],
        # a drawn picture is not a photo, and one picture is not four sides (a small model did both at once)
        [use("make_3d", image="drawn.png", photos=[{"view": v, "file": "drawn.png"} for v in ("front", "left", "back")])],
        lambda m: [use("make_3d", image=files["front"], photos=[{"view": "front", "file": files["side"]},
                                                                {"view": "back", "file": files["side"]}])],
        lambda m: [{"type": "text", "text": "готово"}],
    ])
    studio.providers.pick = lambda: fake
    uploads = [("files", (n, png_bytes(Image.new("RGB", (64, 64), (200, 40, 40))), "image/png")) for n in ("a.png", "b.png")]
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "вот два фото домбры, спереди и слева"}, files=uploads)
    wait_turn(studio, cid)
    results = [b for m in studio.store.messages(cid) if m["role"] == "user" for b in m["content"] if b["type"] == "tool_result"]
    assert len(results) == 5 and all(r["is_error"] for r in results)
    assert "Hunyuan3D-2mv" in results[0]["content"][0]["text"]  # the method was switched by the photos
    assert "Два фото для вида" in results[1]["content"][0]["text"]
    assert "Неверные параметры make_3d" in results[2]["content"][0]["text"]  # «top» is not a card view
    assert "не фото пользователя" in results[3]["content"][0]["text"]
    assert "для нескольких видов" in results[4]["content"][0]["text"]


def test_dombra_goes_through_the_exact_drawing(studio_client):
    """«Домбра» -> the brief knows there is a drawing -> build_exact -> a changed parameter -> the model for Blender."""
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    seen = {}

    def after_brief(messages):
        res = last_result(messages, "write_brief")
        seen["brief_text"] = res["content"][0]["text"]
        return [use("build_exact", template="dombra")]

    def after_first_build(messages):
        res = last_result(messages, "build_exact")
        assert not res["is_error"], res
        seen["table"] = res["content"][0]["text"]
        assert any(c["type"] == "image" for c in res["content"])  # the model sees the preview
        return [use("build_exact", template="dombra", adjust=[{"name": "length", "value": 1.2}, {"name": "frets", "value": 9}])]

    def after_second_build(messages):
        res = last_result(messages, "build_exact")
        seen["second"] = res["ui"]
        return [use("build_exact", template="dombra", adjust=[{"name": "length", "value": 7}])]  # out of range

    def after_bad(messages):
        seen["error"] = last_result(messages, "build_exact")["content"][0]["text"]
        return [{"type": "text", "text": "Готово."}]

    fake = Scripted([
        [use("write_brief", request="домбра для финала шоу")],
        after_brief, after_first_build, after_second_build, after_bad,
    ])
    studio.providers.pick = lambda: fake
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "Домбра для финала шоу"})
    wait_turn(studio, cid, timeout=120)
    assert "ЕСТЬ ТОЧНЫЙ ЧЕРТЁЖ «Домбра» (dombra)" in seen["brief_text"]
    assert "back_facets" in seen["table"] and "Параметры" in seen["table"]  # the model learns the parameter names
    assert seen["second"]["values"]["length"] == 1.2 and seen["second"]["values"]["frets"] == 9
    assert seen["second"]["method"] == "parametric" and seen["second"]["template"] == "dombra"
    assert "от 0.7 до 1.3" in seen["error"]
    chat = client.get(f"/studio/api/chats/{cid}").json()
    meshes = [a for a in chat["assets"] if a["kind"] == "mesh"]
    assert len(meshes) == 2 and meshes[1]["data"]["values"]["length"] == 1.2
    brief = next(a for a in chat["assets"] if a["kind"] == "brief")
    assert brief["data"]["template"] == "dombra" and meshes[0]["parent"] == brief["id"]
    glb = client.get(f"/files/{meshes[1]['files']['mesh']}")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"


def test_sliders_rebuild_the_drawing_in_place(studio_client):
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    first = client.post(f"/studio/api/chats/{cid}/build", json={"template": "dombra", "values": {}}).json()
    assert first["method"] == "parametric" and first["values"]["length"] == 1.0
    second = client.post(f"/studio/api/chats/{cid}/build", json={"template": "dombra", "values": {"length": 1.25}}).json()
    assert second["asset"] == first["asset"] and second["values"]["length"] == 1.25  # replaced, not piled up
    assert second["files"]["mesh"] != first["files"]["mesh"]
    chat = client.get(f"/studio/api/chats/{cid}").json()
    meshes = [a for a in chat["assets"] if a["kind"] == "mesh"]
    assert len(meshes) == 1 and meshes[0]["data"]["values"]["length"] == 1.25 and meshes[0]["data"]["interactive"]
    assert client.get(f"/files/{second['files']['mesh']}").status_code == 200
    bad = client.post(f"/studio/api/chats/{cid}/build", json={"template": "dombra", "values": {"length": 9}})
    assert bad.status_code == 400 and "от 0.7 до 1.3" in bad.json()["detail"]
    assert client.post(f"/studio/api/chats/{cid}/build", json={"template": "unknown"}).status_code == 400
    info = client.get("/studio/api/templates").json()
    assert info["dombra"]["params"]["back_facets"]["integer"] is True


def test_the_brief_is_written_from_what_the_user_typed(studio_client):
    """Not from the assistant's paraphrase with invented checks, and not from «Вариант 1» (an answer): that once turned
    an eagle into «a rectangular block», «the letter A» and a yurt."""
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    requests = []

    class Capturing(Scripted):
        async def structured(self, system, messages, model, effort):
            if model is not ImageBrief:
                return await super().structured(system, messages, model, effort)
            requests.append(messages[0]["content"][0]["text"])
            return BRIEF.model_copy(deep=True)  # write_brief changes the brief it gets: not the shared one

    fake = Capturing([
        [use("write_brief", request="Создать 3D-модель. Проверки: 1 колесо, 7 крыльев")],
        lambda m: [{"type": "text", "text": "бриф готов"}],
        [use("write_brief", request="ещё раз, но другими словами: куб")],
        lambda m: [{"type": "text", "text": "готово"}],
    ])
    studio.providers.pick = lambda: fake
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "Сделай мультяшную машинку с большими колёсами"})
    wait_turn(studio, cid)
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "Вариант 1"})
    wait_turn(studio, cid)
    assert len(requests) == 2
    for text in requests:
        assert "машинку с большими колёсами" in text
        assert "7 крыльев" not in text and "куб" not in text


def test_a_complaint_neither_loses_the_figure_nor_fakes_a_fix(studio_client):
    """«проблемы с текстурой» once became the whole request of a new brief (the eagle was lost), and the same make_3d
    run again was reported as «текстура исправлена»: the brief keeps the figure, a repeated model is called a repeat."""
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    requests, state = [], {}

    class Capturing(Scripted):
        async def structured(self, system, messages, model, effort):
            if model is not ImageBrief:
                return await super().structured(system, messages, model, effort)
            requests.append(messages[0]["content"][0]["text"])
            return BRIEF.model_copy(deep=True)  # write_brief changes the brief it gets: not the shared one

    def first_picture(messages):
        state["image"] = last_result(messages, "generate_images")["ui"]["items"][0]["file"]
        return [use("make_3d", image=state["image"])]

    fake = Capturing([
        [use("write_brief", request="машинка")],
        [use("generate_images", subject=BRIEF.subject, count=1)],
        first_picture,
        lambda m: [{"type": "text", "text": "Модель готова."}],
        lambda m: [use("make_3d", image=state["image"])],
        lambda m: [{"type": "text", "text": "Готово."}],
        [use("write_brief", request="текстура")],
        lambda m: [{"type": "text", "text": "бриф готов"}],
    ])
    studio.providers.pick = lambda: fake
    for text in ("Сделай мультяшную машинку с большими колёсами", "проблемы с текстурой", "цвета на спине неправильные"):
        client.post(f"/studio/api/chats/{cid}/messages", data={"text": text})
        wait_turn(studio, cid)
    results = [b for m in studio.store.messages(cid) if m["role"] == "user" for b in m["content"]
               if b["type"] == "tool_result" and b["ui"].get("kind") == "mesh"]
    every = [(b["ui"].get("kind"), b["content"][0].get("text", "")[:200]) for m in studio.store.messages(cid) if m["role"] == "user" for b in m["content"] if b["type"] == "tool_result"]
    assert len(results) == 2 and not any(r["is_error"] for r in results), every
    assert "повтор" not in results[0]["content"][0]["text"]
    assert "Это повтор" in results[1]["content"][0]["text"] and "Не говори" in results[1]["content"][0]["text"]
    assert len(requests) == 2
    assert "машинку с большими колёсами" in requests[1]  # the figure is still known
    assert "проблемы с текстурой" in requests[1] and "цвета на спине" in requests[1]  # and what the user said since


def test_another_figure_gets_its_own_brief(studio_client):
    """After an eagle «мне нужна лошадь» was carried into the eagle's request, with the eagle's reference sheet, and
    the local model wrote an eagle again: another figure is briefed from the new words alone; a complaint after it
    keeps the horse, not the eagle."""
    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    briefs = []  # (system, request) the prompt engineer got

    class Capturing(Scripted):
        async def structured(self, system, messages, model, effort):
            if model is not ImageBrief:
                return await super().structured(system, messages, model, effort)
            briefs.append((system, messages[0]["content"][0]["text"]))
            return BRIEF.model_copy(deep=True)  # write_brief changes the brief it gets: not the shared one

    fake = Capturing([
        [use("write_brief", request="орёл")], lambda m: [{"type": "text", "text": "бриф готов"}],
        [use("write_brief", request="лошадь")], lambda m: [{"type": "text", "text": "бриф готов"}],
        [use("write_brief", request="текстура")], lambda m: [{"type": "text", "text": "бриф готов"}],
    ], name="local")  # the local model gets only the reference sheets its request names
    studio.providers.pick = lambda: fake
    for text in ("нужен орел с раскрытыми крыльями", "мне нужна лошадь", "проблемы с текстурой"):
        client.post(f"/studio/api/chats/{cid}/messages", data={"text": text})
        wait_turn(studio, cid)
    assert len(briefs) == 3
    (eagle_system, _), (horse_system, horse), (_, complaint) = briefs
    assert '<sheet name="berkut">' in eagle_system and '<sheet name="berkut">' not in horse_system
    assert "лошадь" in horse and "орел" not in horse
    assert "лошадь" in complaint and "проблемы с текстурой" in complaint and "орел" not in complaint
    stored = [a["data"]["request"] for a in studio.store.assets(cid, "brief")]
    assert stored[1] == "мне нужна лошадь"  # the next figure does not inherit the eagle either


def test_photos_marked_with_sides_are_used_once(studio_client):
    """Photos with sides chosen in the app go to the multi-view model; a later message about a drawn picture does not
    reuse them."""
    import io

    from PIL import Image

    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    out = studio.settings.outputs
    Image.new("RGB", (64, 64), (200, 40, 40)).save(out / "drawn.png")

    def png():
        b = io.BytesIO()
        Image.new("RGB", (64, 64), (40, 40, 200)).save(b, "PNG")
        return b.getvalue()

    def first_photo(messages):
        name = next(b["source"] for m in messages if m["role"] == "user" for b in m["content"] if b.get("type") == "image")
        return [use("make_3d", image=name)]

    fake = Scripted([
        first_photo,
        lambda m: [{"type": "text", "text": "готово"}],
        [use("make_3d", image="drawn.png")],
        lambda m: [{"type": "text", "text": "готово"}],
    ])
    studio.providers.pick = lambda: fake
    files = [("files", ("a.png", png(), "image/png")), ("files", ("b.png", png(), "image/png"))]
    r = client.post(f"/studio/api/chats/{cid}/messages", data={"text": "вот фото", "views": '["front", "left"]'}, files=files)
    assert r.status_code == 202, r.text
    wait_turn(studio, cid)
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "а теперь по нарисованной картинке"})
    wait_turn(studio, cid)
    results = [b for m in studio.store.messages(cid) if m["role"] == "user" for b in m["content"] if b["type"] == "tool_result"]
    assert len(results) == 2
    assert "Hunyuan3D-2mv" in results[0]["content"][0]["text"]  # the marked photos -> the multi-view model
    assert "Hunyuan3D-2mv" not in results[1]["content"][0]["text"]  # a new message: the drawn picture, one view
    first = next(m for m in studio.store.messages(cid) if m["role"] == "user")
    assert [b.get("view") for b in first["content"] if b["type"] == "image"] == ["front", "left"]


def test_a_promise_without_a_tool_call_gets_one_reminder(studio_client):
    """The small local model wrote «Теперь сделаю 3D-модель» and stopped: it is reminded once, the reminder is not stored."""
    from dronegen_server.agents.assistant import NUDGE

    client, studio = studio_client
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    cid = client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]
    seen = []

    def after_nudge(messages):
        seen.append(messages[-1]["content"][0]["text"])
        return [{"type": "text", "text": "Теперь сделаю ещё раз."}]  # a second promise: no second reminder

    fake = Scripted([[{"type": "text", "text": "Выбирайте вариант 2. Теперь сделаю 3D-модель."}], after_nudge], name="local")
    studio.providers.pick = lambda: fake
    studio.providers.local = fake
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "беркут"})
    wait_turn(studio, cid)
    assert seen == [NUDGE] and not fake.script
    stored = [b.get("text", "") for m in studio.store.messages(cid) for b in m.get("content", [])]
    assert NUDGE not in stored


def test_file_names_with_words_around_them_are_cleaned():
    """The local model passed «файл 1435_upload.png» (copied from the note next to the photo)."""
    from dronegen_server.agents.tools import clean_name

    assert clean_name("файл 143547024aa0_upload.png") == "143547024aa0_upload.png"
    assert clean_name("«8fd9bf7e6af1_image.png»") == "8fd9bf7e6af1_image.png"
    assert clean_name("f9de14c7b740_mesh.glb") == "f9de14c7b740_mesh.glb"
    assert clean_name("a.png и b.png") == "a.png и b.png"  # two names: not guessed, the tool reports it


# -- looking up a named character -----------------------------------------------------------
def new_chat(client):
    pid = client.post("/studio/api/projects", json={}).json()["id"]
    return client.post("/studio/api/chats", json={"project_id": pid}).json()["id"]


def drain(events):
    out = []
    while not events.empty():
        out.append(events.get_nowait())
    return out


def test_unknown_character_is_looked_up_and_goes_into_the_brief(studio_client, monkeypatch):
    import httpx
    from test_lookup import Wiki

    from dronegen_server.agents import lookup

    client, studio = studio_client
    wiki = Wiki()
    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(wiki))
    cid = new_chat(client)

    def after_look_up(messages):
        res = last_result(messages, "look_up")
        assert not res["is_error"] and "Молния Маккуин" in res["content"][0]["text"]
        assert res["content"][1]["type"] == "image"  # the model sees the picture from the article
        return [use("write_brief", request="Молния Маквин")]

    fake = Scripted([[use("look_up", name="Молния Маквин", english="Lightning McQueen")], after_look_up,
                     [use("look_up", name="молнию маквина", english="")],  # the same character again: no new search
                     [{"type": "text", "text": "Готово."}]])
    studio.providers.pick = lambda: fake
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "сделай молнию маквина"})
    wait_turn(studio, cid)

    chat = client.get(f"/studio/api/chats/{cid}").json()
    ref = next(a for a in chat["assets"] if a["kind"] == "reference")
    assert ref["data"]["source"] == "wikipedia" and ref["data"]["text_en"].startswith("Lightning McQueen")
    assert client.get(f"/files/{ref['files']['image']}").status_code == 200  # the Studio shows the picture
    brief = next(a for a in chat["assets"] if a["kind"] == "brief")
    assert brief["data"]["references"] == ["Молния Маккуин"]
    # the prompt engineer got the English text and the picture before the request
    blocks = fake.briefs[0][0]["content"]
    assert "Lightning McQueen is a red race car." in blocks[0]["text"] and blocks[2]["type"] == "image"
    assert blocks[-1]["text"].startswith("Request:")
    assert len([r for r in wiki.requests if r.url.host == "upload.wikimedia.org"]) == 1  # looked up once
    history_is_valid(chat["messages"])


def test_without_internet_the_user_describes_the_character(studio_client, monkeypatch):
    import httpx

    from dronegen_server.agents import lookup

    def offline(request):
        raise httpx.ConnectError("getaddrinfo failed", request=request)

    client, studio = studio_client
    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(offline))
    cid = new_chat(client)

    def after_answer(messages):
        res = last_result(messages, "look_up")
        text = res["content"][0]["text"]
        assert text.startswith("Справку о «Молния Маквин» найти не удалось (нет связи с интернетом)")
        assert "красная гоночная машинка" in text and "сохранено как справка" in text
        assert res["content"][-1]["type"] == "image"  # the model sees the user's picture
        return [use("write_brief", request="Молния Маквин")]

    fake = Scripted([[{"type": "text", "text": "Узнаю, как он выглядит."}, use("look_up", name="Молния Маквин", english="")],
                     after_answer, [{"type": "text", "text": "Бриф готов."}]])
    studio.providers.pick = lambda: fake
    events = studio.chat.subscribe(cid)
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "сделай молнию маквина"})
    wait_turn(studio, cid)

    seen = drain(events)
    assert any(e["type"] == "notice" and e["level"] == "warn" and "Нет интернета" in e["text"] for e in seen)
    card = next(e for e in seen if e["type"] == "tool_result")
    assert card["ui"] == {"kind": "lookup", "name": "Молния Маквин", "why": "нет связи с интернетом"}
    assert seen[-1]["status"] == "waiting"
    chat = client.get(f"/studio/api/chats/{cid}").json()
    question = chat["pending"]["questions"][0]
    assert question["options"] == ["Опишу словами", "Прикреплю картинку", "Делай как понимаешь"]
    assert question["actions"] == {"Опишу словами": "describe", "Прикреплю картинку": "attach"}
    assert any(m["role"] == "notice" for m in chat["messages"])  # the warning stays in the chat

    files = [("files", ("mcqueen.png", png_bytes(ring_image()), "image/png"))]
    said = "красная гоночная машинка с глазами на лобовом стекле, номер 95"
    assert client.post(f"/studio/api/chats/{cid}/messages", data={"text": said}, files=files).status_code == 202
    wait_turn(studio, cid)

    chat = client.get(f"/studio/api/chats/{cid}").json()
    ref = next(a for a in chat["assets"] if a["kind"] == "reference")
    assert ref["data"]["source"] == "user" and ref["data"]["text"] == said  # without the note about the file
    assert ref["files"]["image"].endswith("_upload.png") and ref["data"]["picture"]
    answer = next(b for m in chat["messages"] if m["role"] == "user" for b in m["content"]
                  if b.get("type") == "tool_result" and b["ui"].get("kind") == "answer")
    assert answer["ui"]["text"] == said and answer["ui"]["lookup"]["why"] == "нет связи с интернетом"
    blocks = fake.briefs[0][0]["content"]
    assert said in blocks[0]["text"] and "from the user" in blocks[0]["text"] and blocks[2]["type"] == "image"
    history_is_valid(chat["messages"])


def test_look_up_switched_off_or_not_needed_never_goes_online(studio_client, monkeypatch):
    import httpx

    from dronegen_server.agents import lookup

    def no_network(request):
        raise AssertionError(f"went online: {request.url}")

    client, studio = studio_client
    monkeypatch.setattr(lookup, "transport", httpx.MockTransport(no_network))
    assert client.get("/studio/api/status").json()["settings"]["web_lookup"] is True  # on by default
    assert client.put("/studio/api/settings", json={"web_lookup": False}).json()["settings"]["web_lookup"] is False

    cid = new_chat(client)
    fake = Scripted([[use("look_up", name="домбра", english="dombra"), use("look_up", name="Губка Боб", english="SpongeBob")],
                     [{"type": "text", "text": "Хорошо, сделаю как понимаю."}]])
    studio.providers.pick = lambda: fake
    client.post(f"/studio/api/chats/{cid}/messages", data={"text": "домбра и губка боб"})
    wait_turn(studio, cid)
    chat = client.get(f"/studio/api/chats/{cid}").json()
    assert "выключен в настройках" in chat["pending"]["questions"][0]["question"]
    known = chat["pending"]["results"][0]  # the dombra has its own reference sheet
    assert known["ui"]["known"] is True and "встроенной справке" in known["content"][0]["text"]

    client.post(f"/studio/api/chats/{cid}/messages", data={"answers": json.dumps([["Делай как понимаешь"]])})
    wait_turn(studio, cid)
    res = last_result(fake.seen[-1], "look_up")
    assert "Делай как понимаешь" in res["content"][0]["text"] and "сохранено" not in res["content"][0]["text"]
    assert not [a for a in studio.store.assets(cid) if a["kind"] == "reference"]  # nothing to keep
    history_is_valid(studio.store.messages(cid))
