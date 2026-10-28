"""Provider-neutral chat history, tools and streaming events.

The history is kept in one format, close to the Anthropic Messages API (a role and a list of
content blocks), with pictures referenced by file name instead of bytes:

    {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "file": "img_ab12.jpg"}]}
    {"role": "assistant", "content": [{"type": "text", "text": "..."},
                                      {"type": "tool_use", "id": "...", "name": "...", "input": {...}}],
     "provider": "claude", "model": "...", "raw": [...]}
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "is_error": false,
                                  "content": [{"type": "text", ...}, {"type": "image", "file": ...}]}]}

``raw`` keeps the provider's own blocks (Claude: thinking blocks with their signatures), so the
same provider gets its turns back unchanged; other providers use ``content``. ``note`` blocks in
``content`` are progress lines for the UI only.

The history only ever grows: earlier messages are never edited, so Claude's thinking blocks stay
valid and its prompt cache keeps hitting (see ``llm.claude``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from pydantic import BaseModel, ValidationError

_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                "maxLength", "minItems", "maxItems", "pattern", "title", "default", "examples"}


def strict_schema(model: type[BaseModel], keep_limits: bool = False) -> dict:
    """JSON schema of a pydantic model in the form structured outputs accept.

    Every object gets ``additionalProperties: false``; constraints the API does not support are
    dropped (pydantic still checks them when the answer is validated). ``keep_limits`` keeps the
    length limits (maxLength, maxItems): llama.cpp turns them into its grammar, so a small model
    cannot loop inside one string until the answer is cut off.
    """
    drop = _UNSUPPORTED - {"maxLength", "maxItems"} if keep_limits else _UNSUPPORTED

    def clean(node):
        if isinstance(node, dict):
            out = {k: clean(v) for k, v in node.items() if k not in drop}
            if out.get("type") == "object" or "properties" in out:
                out["additionalProperties"] = False
                out.setdefault("required", list(out.get("properties", {})))
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(model.model_json_schema())


@dataclass
class Tool:
    """A tool the model may call: the input is validated with ``model`` before it runs."""

    name: str
    description: str
    model: type[BaseModel]

    @property
    def schema(self) -> dict:
        return strict_schema(self.model)

    def parse(self, data) -> BaseModel:
        """The validated input; ValueError with a short reason when it does not fit."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError as e:
                raise ValueError(f"не JSON: {e}") from None
        try:
            return self.model.model_validate(data)
        except ValidationError as e:
            raise ValueError("; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from None


@dataclass
class Event:
    """Something to show while the model answers."""

    kind: str  # text | note | tool_start
    text: str = ""
    data: dict = field(default_factory=dict)


OnEvent = Callable[[Event], Awaitable[None]]


class LLMError(Exception):
    """A readable (Russian) problem with the model. ``offline``: the local model may help."""

    def __init__(self, message: str, offline: bool = False):
        super().__init__(message)
        self.offline = offline


class Provider:
    """One way to talk to a model (Claude API, or the local llama.cpp server)."""

    name = ""
    title = ""
    vision = True

    def available(self) -> tuple[bool, str]:
        return True, ""

    async def respond(self, system: str, messages: list[dict], tools: list[Tool], effort: str,
                      on_event: OnEvent) -> dict:
        """Stream one assistant message; returns it in the neutral format plus ``stop`` and ``usage``."""
        raise NotImplementedError

    async def structured(self, system: str, messages: list[dict], model: type[BaseModel], effort: str) -> BaseModel:
        """One answer that is an instance of ``model`` (structured output)."""
        raise NotImplementedError

    async def text(self, system: str, messages: list[dict], effort: str = "low") -> str:
        """A plain text answer, no tools (a model describing what it sees before it counts)."""

        async def ignore(_event: Event) -> None:
            pass

        return text_of(await self.respond(system, messages, [], effort, ignore))

    async def close(self) -> None:
        pass


def text_of(message: dict) -> str:
    """The visible text of a neutral message."""
    return "".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text")


def tool_uses(message: dict) -> list[dict]:
    return [b for b in message.get("content", []) if b.get("type") == "tool_use"]
