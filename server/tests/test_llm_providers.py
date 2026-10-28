"""Claude and local providers without network: request shapes, streaming, errors."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from dronegen_server.agents.tools import TOOLS, AskUser
from dronegen_server.llm import Event, LLMError, strict_schema
from dronegen_server.llm import images as llm_images
from dronegen_server.llm.claude import BETAS, Claude, _echo
from dronegen_server.llm.local import Local

anthropic = pytest.importorskip("anthropic")


@pytest.fixture()
def picture(tmp_path):
    return llm_images.add(tmp_path, Image.new("RGB", (64, 32), (200, 30, 30)))


def history(picture):
    return [
        {"role": "user", "content": [{"type": "image", "file": picture, "source": "x.png"}, {"type": "text", "text": "домбра"}]},
        {"role": "assistant", "provider": "claude",
         "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "make_3d", "input": {"image": "x.png"}}],
         "raw": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "ok"},
                 {"type": "tool_use", "id": "t1", "name": "make_3d", "input": {"image": "x.png"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False, "ui": {"kind": "mesh"},
                                      "content": [{"type": "text", "text": "готово"}, {"type": "image", "file": picture}]}]},
        {"role": "assistant", "provider": "local", "content": [{"type": "note", "text": "думаю"}, {"type": "text", "text": ""},
                                                               {"type": "tool_use", "id": "t2", "name": "ask_user", "input": {}}]},
    ]


def test_images_are_stored_once_by_content(tmp_path):
    a = llm_images.add(tmp_path, Image.new("RGBA", (3000, 1000), (0, 0, 0, 0)))
    b = llm_images.add(tmp_path, Image.new("RGBA", (3000, 1000), (0, 0, 0, 0)))
    assert a == b and Image.open(tmp_path / a).size == (1024, 341)
    with pytest.raises(ValueError):
        llm_images.path(tmp_path, "../secrets.json")


def test_strict_schema_closes_every_object():
    s = strict_schema(AskUser)
    assert s["additionalProperties"] is False
    q = s["$defs"]["Question"]
    assert q["additionalProperties"] is False and "maxItems" not in json.dumps(s)
    assert set(s["required"]) == {"questions"}
    assert {t.name for t in TOOLS} == {"ask_user", "look_up", "write_brief", "generate_images", "make_3d", "build_exact", "send_to_blender"}
    exact = next(t for t in TOOLS if t.name == "build_exact").schema
    assert exact["properties"]["template"]["enum"] == ["baiterek", "dombra", "shanyrak", "yurt"] or         set(exact["properties"]["template"]["enum"]) == {"dombra", "shanyrak", "yurt", "baiterek"}
    assert exact["additionalProperties"] is False


def test_claude_request_from_history(tmp_path, picture):
    c = Claude(lambda: "key", tmp_path)
    api = c.to_api(history(picture))
    assert api[0]["content"][0]["source"]["media_type"] == "image/jpeg" and api[0]["content"][0]["source"]["data"]
    assert api[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}  # Claude's own turn verbatim
    result = api[2]["content"][0]
    assert set(result) == {"type", "tool_use_id", "is_error", "content"} and result["content"][1]["type"] == "image"
    assert api[3]["content"] == [{"type": "tool_use", "id": "t2", "name": "ask_user", "input": {}}]  # no note, no empty text


def test_echo_drops_the_declined_part_before_a_fallback():
    raw = [{"type": "thinking", "thinking": "a"}, {"type": "text", "text": "начало"}, {"type": "tool_use", "id": "x"},
           {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
           {"type": "thinking", "thinking": "b"}, {"type": "text", "text": "ответ"}]
    assert _echo(raw) == [{"type": "text", "text": "начало"}, {"type": "thinking", "thinking": "b"},
                          {"type": "text", "text": "ответ"}]
    assert _echo(raw[:2]) == raw[:2]


class Block(NS):
    def to_dict(self):
        return {k: v for k, v in vars(self).items()}


class FakeStream:
    def __init__(self, events, final, error=None):
        self.events, self.final, self.error = events, final, error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()

    async def get_final_message(self):
        return self.final


def fake_client(stream, calls):
    def stream_fn(**params):
        calls.append(params)
        return stream

    return NS(beta=NS(messages=NS(stream=stream_fn)))


def test_claude_streams_and_returns_the_neutral_message(tmp_path, picture, monkeypatch):
    events = [
        NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking="Смотрю картинку")),
        NS(type="content_block_delta", index=1, delta=NS(type="text_delta", text="Делаю ")),
        NS(type="content_block_start", index=2, content_block=NS(type="tool_use", id="tu1", name="make_3d")),
    ]
    final = NS(content=[Block(type="thinking", thinking="Смотрю картинку", signature="s1"),
                        Block(type="text", text="Делаю 3D", citations=None),
                        Block(type="tool_use", id="tu1", name="make_3d", input={"image": "a.png", "method": "hunyuan"})],
               usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=7, cache_creation_input_tokens=0,
                        iterations=[NS(type="message")]),
               stop_reason="tool_use", model="claude-opus-5-5", stop_details=None, input_transformations=[])
    calls, seen = [], []
    c = Claude(lambda: "key", tmp_path)
    monkeypatch.setattr(c, "client", lambda: fake_client(FakeStream(events, final), calls))

    async def on_event(e: Event):
        seen.append((e.kind, e.text))

    msg = asyncio.run(c.respond("system", history(picture), TOOLS, "low", on_event))
    assert seen == [("note", "Смотрю картинку"), ("text", "Делаю "), ("tool_start", "")]
    assert msg["content"] == [{"type": "note", "text": "Смотрю картинку"}, {"type": "text", "text": "Делаю 3D"},
                              {"type": "tool_use", "id": "tu1", "name": "make_3d", "input": {"image": "a.png", "method": "hunyuan"}}]
    assert msg["raw"][1] == {"type": "text", "text": "Делаю 3D"}  # None fields are not echoed
    assert msg["stop"] == "tool_use" and msg["usage"]["cache_read"] == 7 and msg["fallback"] is False
    p = calls[0]
    assert p["model"] == "claude-opus-5-5" and p["betas"] == BETAS and p["fallbacks"] == "default"
    assert p["thinking"] == {"type": "adaptive", "display": "updates", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert p["output_config"] == {"effort": "low"} and p["cache_control"] == {"type": "ephemeral"}
    assert all(t["eager_input_streaming"] for t in p["tools"]) and "tool_choice" not in p


def test_claude_errors_are_readable(tmp_path, monkeypatch):
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = [
        (anthropic.APIConnectionError(request=req), "Нет связи", True),
        (anthropic.AuthenticationError("bad", response=httpx2.Response(401, request=req), body=None), "Ключ", True),
        (anthropic.BadRequestError("too long", response=httpx2.Response(400, request=req), body=None), "отклонил", False),
    ]
    c = Claude(lambda: "key", tmp_path)
    for error, text, offline in cases:
        monkeypatch.setattr(c, "client", lambda e=error: fake_client(FakeStream([], None, e), []))
        with pytest.raises(LLMError) as info:
            asyncio.run(c.respond("s", [], [], "low", lambda e: asyncio.sleep(0)))
        assert text in str(info.value) and info.value.offline is offline
    assert Claude(lambda: "", tmp_path).available()[0] is False


def test_local_request_from_history(tmp_path, picture):
    loc = Local(server=None, image_dir=tmp_path)
    api = loc.to_api("system", history(picture))
    roles = [m["role"] for m in api]
    assert roles == ["system", "user", "assistant", "tool", "user", "assistant"]
    assert api[1]["content"][0]["type"] == "image_url"
    assert api[2]["tool_calls"][0]["function"] == {"name": "make_3d", "arguments": '{"image": "x.png"}'}
    assert api[3] == {"role": "tool", "tool_call_id": "t1", "content": "готово"}
    assert api[4]["content"][1]["type"] == "image_url"  # the tool's picture follows as a user message
    files = [llm_images.add(tmp_path, Image.new("RGB", (16, 16), (i * 20, 0, 0))) for i in range(10)]
    many = [{"role": "user", "content": [{"type": "image", "file": f}]} for f in files]
    parts = [p for m in loc.to_api("s", many)[1:] for p in m["content"]]
    assert sum(p["type"] == "image_url" for p in parts) == 6  # older pictures become a mention


def sse(*chunks):
    return "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_local_streams_text_and_tool_calls(tmp_path):
    httpx = pytest.importorskip("httpx")
    body = sse(
        {"choices": [{"delta": {"content": "Сейчас "}}]},
        {"choices": [{"delta": {"content": "сделаю."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "write_brief", "arguments": '{"req'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'uest": "домбра"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
    )
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    loc = Local(server=NS(url="http://llm", running=lambda: True), image_dir=tmp_path)
    loc.transport = httpx.MockTransport(handler)
    seen = []

    async def on_event(e):
        seen.append(e.kind)

    msg = asyncio.run(loc.respond("system", [{"role": "user", "content": [{"type": "text", "text": "домбра"}]}],
                                  TOOLS, "low", on_event))
    assert msg["content"] == [{"type": "text", "text": "Сейчас сделаю."},
                              {"type": "tool_use", "id": "c1", "name": "write_brief", "input": {"request": "домбра"}}]
    assert msg["stop"] == "tool_use" and msg["usage"] == {"input": 50, "output": 9}
    assert seen.count("text") == 2 and "tool_start" in seen
    assert sent[0]["tools"][0]["function"]["name"] == "ask_user" and sent[0]["stream"] is True


def test_local_structured_answer(tmp_path):
    httpx = pytest.importorskip("httpx")
    from dronegen_server.agents.prompt_engineer import ImageBrief

    brief = {"title_ru": "Домбра", "subject": "a dombra", "view": "front view", "style": "sheet", "method": "hunyuan_mv",
             "symmetry": True, "checks": ["2 струны"], "counts": [{"part": "колки", "expected": 2}], "negative": "pipa",
             "notes_ru": "Лист видов."}
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(brief)}}]})

    loc = Local(server=NS(url="http://llm", running=lambda: True), image_dir=tmp_path)
    loc.transport = httpx.MockTransport(handler)
    out = asyncio.run(loc.structured("s", [{"role": "user", "content": [{"type": "text", "text": "домбра"}]}], ImageBrief, "medium"))
    assert out.method == "hunyuan_mv" and out.prompt().startswith("character model sheet")
    assert sent[0]["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
