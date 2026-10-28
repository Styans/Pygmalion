"""Claude API through the official ``anthropic`` SDK: the default brain of the Studio chat.

* Model ``claude-opus-5-5`` (changeable in the Studio settings). Its thinking is always on;
  ``effort`` sets how much it thinks: ``low`` in the chat, ``medium`` for briefs.
* Refusal fallback is on: ``fallbacks="default"`` (beta ``server-side-fallback-2026-07-01``). A
  request Claude's safety classifiers decline is re-run server-side on the model Anthropic
  recommends for that case, instead of failing the chat.
* The history only grows and Claude's own turns go back verbatim (``raw``, with thinking blocks).
  ``prefix_mismatch_behavior: "drop_block"`` (beta ``thinking-binding-controls-2026-08-01``) keeps
  old chats working after an update changes the system prompt or the tools: their old reasoning
  is dropped instead of the request failing.
* Notes the model writes between tool calls arrive as thinking blocks; ``display: "updates"``
  (beta ``thinking-display-updates-2026-08-18``) returns their text for the status line.
* Tool inputs stream as generated (``eager_input_streaming``) and are validated by the caller.
* ``cache_control`` on the request caches the whole prefix, so each turn re-reads it cheaply.

The tool loop itself is in ``agents.assistant`` (not the SDK's tool runner): the same loop drives
the local model, and a turn can stop to wait for the user's answer between HTTP requests.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from . import images
from .base import Event, LLMError, OnEvent, Provider, Tool, strict_schema

log = logging.getLogger("dronegen.llm.claude")

MODEL = "claude-opus-5-5"
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")
BETAS = ["server-side-fallback-2026-07-01", "thinking-binding-controls-2026-08-01", "thinking-display-updates-2026-08-18"]
MAX_TOKENS = 16000  # thinking counts towards it as well


def _echo(raw: list[dict]) -> list[dict]:
    """Blocks to send back next time. After a mid-answer fallback the declined model's
    thinking and tool calls before the last ``fallback`` marker are left out (API rule)."""
    marks = [i for i, b in enumerate(raw) if b.get("type") == "fallback"]
    if not marks:
        return raw
    cut = marks[-1]
    head = [b for b in raw[:cut] if b.get("type") not in ("thinking", "redacted_thinking", "tool_use", "fallback")]
    return head + raw[cut + 1:]


class Claude(Provider):
    name = "claude"
    title = "Claude"

    def __init__(self, api_key: Callable[[], str], image_dir: Path, model: Callable[[], str] = lambda: MODEL):
        self._key = api_key
        self._model = model
        self.image_dir = image_dir
        self._client = None
        self._client_key = ""

    @property
    def model(self) -> str:
        return self._model() or MODEL

    def available(self):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "не установлен пакет anthropic — запустите python install.py"
        return (True, "") if self._key() else (False, "не задан ключ Claude API (настройки студии)")

    def client(self):
        import anthropic

        key = self._key()
        if not key:
            raise LLMError("Не задан ключ Claude API: откройте настройки студии", offline=True)
        if self._client is None or key != self._client_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=key, max_retries=2, timeout=anthropic.Timeout(600.0, connect=15.0)
            )
            self._client_key = key
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- history -> request -------------------------------------------------------------
    def _block(self, b: dict) -> dict | None:
        t = b["type"]
        if t == "text":
            return {"type": "text", "text": b["text"]} if b.get("text") else None
        if t == "image":
            data = images.b64(self.image_dir, b["file"])
            return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
        if t == "tool_use":
            return {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}}
        if t == "tool_result":
            inner = [x for x in (self._block(c) for c in b.get("content", [])) if x]
            return {"type": "tool_result", "tool_use_id": b["tool_use_id"], "is_error": bool(b.get("is_error")),
                    "content": inner or [{"type": "text", "text": "(пусто)"}]}
        return None  # notes and anything else are for the UI only

    def to_api(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] not in ("user", "assistant"):
                continue  # notices are for the people reading the chat
            if m["role"] == "assistant" and m.get("provider") == "claude" and m.get("raw"):
                out.append({"role": "assistant", "content": m["raw"]})
                continue
            blocks = [x for x in (self._block(b) for b in m.get("content", [])) if x]
            if blocks:
                out.append({"role": m["role"], "content": blocks})
        return out

    # -- requests -----------------------------------------------------------------------
    @asynccontextmanager
    async def _errors(self):
        import anthropic

        try:
            yield
        except LLMError:
            raise
        except anthropic.AuthenticationError:
            raise LLMError("Ключ Claude API не подходит — проверьте его в настройках студии", offline=True) from None
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"У ключа Claude API нет доступа к модели {self.model}: {e.message}", offline=True) from None
        except anthropic.NotFoundError:
            raise LLMError(f"Модель {self.model} недоступна для этого ключа", offline=True) from None
        except anthropic.RateLimitError:
            raise LLMError("Claude API: слишком много запросов, подождите минуту", offline=True) from None
        except anthropic.APIConnectionError:
            raise LLMError("Нет связи с Claude API (проверьте интернет)", offline=True) from None
        except anthropic.BadRequestError as e:
            raise LLMError(f"Claude API отклонил запрос: {e.message}") from None
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMError("Claude API временно перегружен, попробуйте позже", offline=True) from None
            raise LLMError(f"Ошибка Claude API ({e.status_code}): {e.message}") from None

    async def respond(self, system: str, messages: list[dict], tools: list[Tool], effort: str, on_event: OnEvent) -> dict:
        params = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            betas=BETAS,
            fallbacks="default",
            system=[{"type": "text", "text": system}],
            thinking={"type": "adaptive", "display": "updates", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
            output_config={"effort": effort},
            tools=[{"name": t.name, "description": t.description, "input_schema": t.schema, "eager_input_streaming": True}
                   for t in tools],
            messages=self.to_api(messages),
            cache_control={"type": "ephemeral"},
        )
        async with self._errors():
            for attempt in range(3):
                try:
                    final = await self._stream(params, on_event)
                    break
                except ValueError:  # a tool input the SDK could not parse at all: ask again
                    if attempt == 2:
                        raise LLMError("Claude трижды прислал повреждённый вызов инструмента") from None
                    log.warning("unparseable tool input from Claude, re-asking")
                    await on_event(Event("reset"))
        raw = _echo([{k: v for k, v in b.to_dict().items() if v is not None} for b in final.content])
        content = []
        for b in raw:
            if b["type"] == "text" and b.get("text"):
                content.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                content.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}})
            elif b["type"] == "thinking" and b.get("thinking"):
                content.append({"type": "note", "text": b["thinking"]})
        usage = final.usage
        fallback = any(getattr(it, "type", "") == "fallback_message" for it in (getattr(usage, "iterations", None) or []))
        details = getattr(final, "stop_details", None)
        dropped = getattr(final, "input_transformations", None)
        if dropped:
            log.info("thinking blocks dropped by the API: %s", [d.to_dict() for d in dropped])
        return {
            "role": "assistant",
            "content": content,
            "raw": raw,
            "provider": self.name,
            "model": final.model,
            "stop": final.stop_reason,
            "stop_category": getattr(details, "category", None) if details else None,
            "fallback": fallback,
            "usage": {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            },
        }

    async def _stream(self, params: dict, on_event: OnEvent):
        async with self.client().beta.messages.stream(**params) as stream:
            async for ev in stream:
                if ev.type == "content_block_start" and ev.content_block.type == "tool_use":
                    await on_event(Event("tool_start", data={"id": ev.content_block.id, "name": ev.content_block.name}))
                elif ev.type == "content_block_delta":
                    d = ev.delta
                    if d.type == "text_delta" and d.text:
                        await on_event(Event("text", d.text, {"index": ev.index}))
                    elif d.type == "thinking_delta" and d.thinking:
                        await on_event(Event("note", d.thinking, {"index": ev.index}))
            return await stream.get_final_message()

    async def structured(self, system: str, messages: list[dict], model: type[BaseModel], effort: str) -> BaseModel:
        params = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[{"type": "text", "text": system}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": strict_schema(model)}},
            messages=self.to_api(messages),
            cache_control={"type": "ephemeral"},
        )
        async with self._errors():
            async with self.client().beta.messages.stream(**params) as stream:
                final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            raise LLMError("Claude отказался выполнить этот запрос")
        if final.stop_reason == "max_tokens":
            raise LLMError("Ответ Claude не поместился в лимит — попробуйте упростить запрос")
        text = next((b.text for b in final.content if b.type == "text"), "")
        try:
            return model.model_validate_json(text)
        except ValidationError as e:
            raise LLMError(f"Claude вернул ответ не по схеме: {e.errors()[0]['msg']}") from None
