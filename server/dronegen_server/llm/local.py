"""The local model: llama.cpp's ``llama-server`` with Qwen3-VL-8B-Instruct (Apache-2.0).

It works without internet and sees pictures (it counts strings and pegs on a generated dombra).
The server process is started on demand (~10 s) on 127.0.0.1 and takes ~6 GB of video memory, so
it cannot share an 8 GB card with Z-Image or Hunyuan3D: it is stopped before every GPU job
(``JobManager.gpu_hooks``), and the GPU models are unloaded before it starts again.

Measured on an RTX 3050 8 GB: 24 tokens/s, a tool call in ~2 s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import subprocess
import sys
import threading
import time

from pydantic import BaseModel, ValidationError

from .. import weights
from . import images
from .base import Event, LLMError, OnEvent, Provider, Tool, strict_schema

log = logging.getLogger("dronegen.llm.local")

MODEL_FILE = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
CTX = 16384
IMAGE_SIDE = 768  # ~580 image tokens each; small details (pegs) need it
MAX_IMAGES = 6  # older pictures in the history become a text mention (the context is small)
HISTORY_CHARS = 30000  # ~10k tokens of text: older messages are left out


def _size(msg: dict) -> int:
    """Rough length of a message in characters; a picture counts as ~1000."""
    content = msg.get("content") or ""
    if isinstance(content, str):
        n = len(content)
    else:
        n = sum(len(p.get("text", "")) if p["type"] == "text" else 1000 for p in content)
    return n + sum(len(c["function"]["arguments"]) for c in msg.get("tool_calls", []))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_job = None


def _bind_to_this_process(proc: subprocess.Popen) -> None:
    """Windows: put the child into a job object that kills it when this process ends, however it ends.

    Without it a server closed with the window's X (or crashed) leaves llama-server holding ~6 GB of
    video memory until the next start.
    """
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if _job is None:
        _job = kernel32.CreateJobObjectW(None, None)

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(_job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimitInformation
    handle = kernel32.OpenProcess(0x1F0FFF, False, proc.pid)  # PROCESS_ALL_ACCESS
    if handle:
        kernel32.AssignProcessToJobObject(_job, handle)
        kernel32.CloseHandle(handle)


class LlamaServer:
    """One ``llama-server`` process, started on demand and stopped for GPU jobs."""

    def __init__(self, settings):
        self.settings = settings
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self._lock = threading.Lock()

    def files(self):
        folder = weights.folder(self.settings, "qwen3vl").resolve()  # llama-server runs in its own folder
        exe = weights.llama_server(self.settings)
        return (exe.resolve() if exe else None), folder / MODEL_FILE, folder / MMPROJ_FILE

    def available(self) -> tuple[bool, str]:
        missing = weights.missing(self.settings, "llm")
        if missing:
            return False, weights.missing_message(missing)
        exe, model, mmproj = self.files()
        if exe is None or not model.is_file() or not mmproj.is_file():
            return False, "файлы локальной модели не найдены — запустите python install.py"
        return True, ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _kill_strays(self, exe) -> None:
        """A llama-server of ours left over from a crashed run still holds video memory."""
        import psutil

        for p in psutil.process_iter(["exe"]):
            try:
                if p.info["exe"] and p.info["exe"].lower() == str(exe).lower() and p.pid != getattr(self.proc, "pid", None):
                    log.info("stopping a stray llama-server (pid %s)", p.pid)
                    p.kill()
            except Exception:  # noqa: BLE001 - gone or not ours to inspect
                pass

    def start(self, timeout: float = 180.0) -> None:
        """Start the server (if needed) and wait until the model is loaded. Blocking."""
        with self._lock:
            if self.running():
                return
            ok, reason = self.available()
            if not ok:
                raise LLMError(f"Локальная модель недоступна: {reason}")
            exe, model, mmproj = self.files()
            self._kill_strays(exe)
            self.port = _free_port()
            cmd = [str(exe), "-m", str(model), "--mmproj", str(mmproj), "-c", str(CTX), "-ngl", "99", "-fa", "on",
                   "--host", "127.0.0.1", "--port", str(self.port), "--no-webui", "-np", "1", "--jinja"]
            self.settings.logs.mkdir(parents=True, exist_ok=True)
            logfile = open(self.settings.logs / "llama-server.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, cwd=exe.parent, creationflags=flags)
            logfile.close()
            try:
                _bind_to_this_process(self.proc)
            except Exception:  # noqa: BLE001 - only a safety net
                log.warning("could not bind llama-server to this process", exc_info=True)
        log.info("llama-server starting on port %s", self.port)
        t0 = time.time()
        import urllib.request

        while time.time() - t0 < timeout:
            if not self.running():
                tail = (self.settings.logs / "llama-server.log").read_text(encoding="utf-8", errors="replace")[-900:]
                raise LLMError("Локальная модель не запустилась (возможно, мало видеопамяти).\n"
                               "Конец журнала llama-server:\n" + tail)
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if b'"ok"' in r.read():
                        log.info("llama-server ready in %.1f s", time.time() - t0)
                        return
            except OSError:
                pass
            time.sleep(0.5)
        self.stop()
        raise LLMError("Локальная модель не загрузилась за 3 минуты")

    def stop(self) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                log.info("stopping llama-server")
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None


class Local(Provider):
    name = "local"
    title = "Локальная модель (Qwen3-VL-8B)"
    transport = None  # httpx transport override (tests)

    def __init__(self, server: LlamaServer, image_dir, make_room=None):
        self.server = server
        self.image_dir = image_dir
        self.make_room = make_room  # blocking callable: free the video memory (JobManager.release_gpu)

    def available(self):
        return self.server.available()

    async def ready(self) -> None:
        if self.server.running():
            return
        if self.make_room is not None and not await asyncio.to_thread(self.make_room):
            raise LLMError("Видеокарта занята задачами — локальная модель запустится, когда они закончатся")
        await asyncio.to_thread(self.server.start)

    # -- history -> OpenAI messages -----------------------------------------------------
    def _images_to_keep(self, messages: list[dict]) -> set[str]:
        files = []
        for m in messages:
            for b in m.get("content", []):
                if b.get("type") == "image":
                    files.append(b["file"])
                for c in b.get("content", []) if b.get("type") == "tool_result" else []:
                    if c.get("type") == "image":
                        files.append(c["file"])
        return set(files[-MAX_IMAGES:])

    def _parts(self, blocks: list[dict], keep: set[str]) -> list[dict]:
        parts = []
        for b in blocks:
            if b["type"] == "text" and b.get("text"):
                parts.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image":
                if b["file"] in keep:
                    data = images.smaller_b64(self.image_dir, b["file"], IMAGE_SIDE)
                    parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}})
                else:
                    parts.append({"type": "text", "text": f"[картинка {b['file']}]"})
        return parts

    def to_api(self, system: str, messages: list[dict]) -> list[dict]:
        keep = self._images_to_keep(messages)
        out = []
        for m in messages:
            if m["role"] not in ("user", "assistant"):
                continue  # notices are for the people reading the chat
            if m["role"] == "assistant":
                text = "".join(b.get("text", "") for b in m["content"] if b["type"] == "text")
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                         for b in m["content"] if b["type"] == "tool_use"]
                msg = {"role": "assistant", "content": text}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
                continue
            plain = [b for b in m["content"] if b["type"] != "tool_result"]
            pictures = []
            for b in m["content"]:
                if b["type"] != "tool_result":
                    continue
                parts = self._parts(b.get("content", []), keep)
                text = "\n".join(p["text"] for p in parts if p["type"] == "text") or "(пусто)"
                if b.get("is_error"):
                    text = "ОШИБКА: " + text
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": text})
                pictures += [p for p in parts if p["type"] == "image_url"]
            if pictures:  # the chat template takes pictures only in user messages
                out.append({"role": "user", "content": [{"type": "text", "text": "Картинки из результата инструмента:"}, *pictures]})
            if plain:
                out.append({"role": "user", "content": self._parts(plain, keep) or [{"type": "text", "text": "…"}]})
        # the context is small: drop the oldest turns (never leaving a tool result without its call)
        while sum(_size(x) for x in out) > HISTORY_CHARS and len(out) > 2:
            out.pop(0)
            while out and out[0]["role"] == "tool":
                out.pop(0)
        return [{"role": "system", "content": system}, *out]

    # -- requests -----------------------------------------------------------------------
    async def _post(self, body: dict, stream: bool):
        import httpx

        await self.ready()
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), transport=self.transport)
        try:
            req = client.build_request("POST", self.server.url + "/v1/chat/completions", json=body)
            resp = await client.send(req, stream=stream)
        except httpx.HTTPError as e:
            await client.aclose()
            raise LLMError(f"Локальная модель не отвечает: {e}") from None
        if resp.status_code != 200:
            text = (await resp.aread()).decode("utf-8", "replace")[:400]
            await resp.aclose()
            await client.aclose()
            raise LLMError(f"Локальная модель вернула ошибку {resp.status_code}: {text}")
        return client, resp

    async def respond(self, system: str, messages: list[dict], tools: list[Tool], effort: str, on_event: OnEvent) -> dict:
        body = {
            "messages": self.to_api(system, messages),
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
                      for t in tools],
            "stream": True,
            "max_tokens": 2048,
            "temperature": 0.4,
        }
        client, resp = await self._post(body, stream=True)
        text, calls, stop, usage = [], {}, "end_turn", {}
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text.append(delta["content"])
                        await on_event(Event("text", delta["content"], {"index": 0}))
                    for tc in delta.get("tool_calls") or []:
                        acc = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        fn = tc.get("function") or {}
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                            await on_event(Event("tool_start", data={"id": acc["id"], "name": acc["name"]}))
                        acc["args"] += fn.get("arguments") or ""
                    reason = choice.get("finish_reason")
                    if reason:
                        stop = {"tool_calls": "tool_use", "length": "max_tokens"}.get(reason, "end_turn")
        except Exception as e:  # noqa: BLE001 - a broken stream (the server was stopped for a GPU job)
            raise LLMError(f"Ответ локальной модели прервался: {e}") from None
        finally:
            await resp.aclose()
            await client.aclose()
        content = [{"type": "text", "text": "".join(text)}] if "".join(text).strip() else []
        for i, acc in sorted(calls.items()):
            try:
                args = json.loads(acc["args"] or "{}")
            except ValueError:
                args = {"__invalid_json__": acc["args"]}
            content.append({"type": "tool_use", "id": acc["id"] or f"call_{int(time.time() * 1000)}_{i}",
                            "name": acc["name"], "input": args})
        if calls and stop == "end_turn":
            stop = "tool_use"
        return {"role": "assistant", "content": content, "provider": self.name, "model": MODEL_FILE, "stop": stop,
                "usage": {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)}}

    async def structured(self, system: str, messages: list[dict], model: type[BaseModel], effort: str) -> BaseModel:
        """A JSON answer by the schema. A small model sometimes loops inside one string and never closes the
        JSON: the second try is livelier (higher temperature, a penalty for repeating) and shorter."""
        last = ""
        for attempt in range(2):
            body = {
                "messages": self.to_api(system, messages),
                "response_format": {"type": "json_schema", "json_schema": {"name": model.__name__, "strict": True,
                                                                           "schema": strict_schema(model, keep_limits=True)}},
                "max_tokens": 2048 if attempt == 0 else 1400,
                "temperature": 0.3 if attempt == 0 else 0.8,
            }
            if attempt:
                body["repeat_penalty"] = 1.25
            client, resp = await self._post(body, stream=False)
            try:
                data = json.loads(await resp.aread())
            finally:
                await resp.aclose()
                await client.aclose()
            text = data["choices"][0]["message"].get("content") or ""
            try:
                return model.model_validate_json(text)
            except ValidationError as e:
                last = e.errors()[0]["msg"]
                log.warning("Локальная модель ответила не по схеме (попытка %d): %s", attempt + 1, last)
        raise LLMError(f"Локальная модель ответила не по схеме: {last}") from None
