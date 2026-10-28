"""Chat models. Everything runs locally by default: Qwen3-VL-8B through llama.cpp on this computer.

``Settings → provider``: ``local`` (the default; nothing leaves the computer) or ``claude`` (the
Claude API in the cloud: only when the user switches to it and enters a key; when Claude cannot be
reached, the turn continues on the local model). An old ``auto`` setting means ``local``.
"""

from __future__ import annotations

from .base import Event, LLMError, Provider, Tool, strict_schema, text_of, tool_uses  # noqa: F401
from .claude import Claude
from .local import Local, LlamaServer

MODES = ("local", "claude")


class Providers:
    def __init__(self, settings, config, manager=None):
        self.config = config
        self.image_dir = settings.home / "studio" / "llm"
        self.claude = Claude(lambda: config.secret("anthropic_api_key"), self.image_dir,
                             lambda: config.get("claude_model"))
        self.server = LlamaServer(settings)
        self.local = Local(self.server, self.image_dir, make_room=manager.release_gpu if manager else None)
        if manager is not None:
            manager.gpu_hooks.append(self.server.stop)

    @property
    def mode(self) -> str:
        mode = self.config.get("provider")
        return mode if mode in MODES else "local"

    def pick(self) -> Provider:
        if self.mode == "claude" and self.claude.available()[0]:
            return self.claude
        return self.local

    def fallback(self, current: Provider) -> Provider | None:
        """The local model, when the chosen cloud model cannot be reached."""
        if current is self.claude and self.local.available()[0]:
            return self.local
        return None

    def status(self) -> dict:
        c_ok, c_reason = self.claude.available()
        l_ok, l_reason = self.local.available()
        current = self.pick()
        return {
            "mode": self.mode,
            "current": current.name,
            "cloud": current is self.claude,
            "fallback": self.mode == "claude" and current is self.local,  # the cloud was chosen but cannot be used
            "claude": {"available": c_ok, "reason": c_reason, "model": self.claude.model},
            "local": {"available": l_ok, "reason": l_reason, "running": self.server.running()},
        }

    async def close(self) -> None:
        await self.claude.close()
        self.server.stop()
