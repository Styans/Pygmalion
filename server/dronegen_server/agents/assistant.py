"""The chat assistant: one turn = the model answers, runs tools, looks at the results, answers again.

A turn runs as a background task per chat; everything it does is published as events (the
Studio listens over SSE) and every finished message is stored, so a reloaded page shows the same
chat. When the model asks the user something (``ask_user``), or a tool cannot go on without the
user (``look_up`` without internet: «как выглядит Молния Маквин?»), the turn stops; the user's
answer becomes that tool's result and the next turn continues from there. A description or a
picture the user gives in answer to ``look_up`` is kept as the thing's reference for the brief.

History rules (``llm.base``): messages are only appended; a stopped or failed turn is closed
with error results for its open tool calls, so the history always stays valid for the API.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from PIL import Image

from ..llm import LLMError, Providers, tool_uses
from ..llm import images as llm_images
from . import tools as T

log = logging.getLogger("dronegen.agents")

MAX_STEPS = 16  # model calls per turn
# «Теперь сделаю 3D-модель» and the turn ends: the small local model says it and does not do it
PROMISE = re.compile(r"\b(сделаю|делаю|создам|создаю|построю|строю|нарисую|рисую|сгенерирую|генерирую|запускаю|запущу)\b",
                     re.IGNORECASE)
NUDGE = ("(Служебное сообщение программы, не от пользователя.) Ты написал, что сделаешь это, но не вызвал инструмент. "
         "Вызови нужный инструмент сейчас. Если нужно решение пользователя, вызови ask_user.")
EFFORT = "low"

SYSTEM = """\
Ты — ассистент DroneGen Studio. Помогаешь художнику дрон-шоу придумать фигуру и получить её 3D-модель \
для Blender и Skybrush. Сотни дронов рисуют фигуру в ночном небе, её видно издалека, поэтому нужны \
чёткий силуэт и крупные узнаваемые части. Многие фигуры — казахские мотивы (домбра, тулпар, шанырак, беркут).

Как общаться
- Пиши по-русски, просто и коротко: обычно 1-3 предложения. Без канцелярита и лишних английских слов.
- Если запрос можно понять по-разному и от этого зависит результат (стиль, поза, вид, цвета, детализация), \
задай 1-3 коротких вопроса инструментом ask_user с вариантами ответа. О том, что можно разумно выбрать \
самому, не спрашивай.
- Когда всё ясно, одной фразой скажи, что делаешь, и действуй.
- Пиши о результате, а не пересказывай шаги. Не повторяй то, что пользователь и так видит на картинках \
и в карточках чата.

Как делать фигуру
0. look_up — если в запросе назван конкретный персонаж (мультфильм, фильм, игра, книга), бренд или логотип, \
известное здание или человек (Молния Маквин, Губка Боб, логотип Apple, Эйфелева башня), сначала узнай, как \
он выглядит, даже если кажется, что знаешь. Для обычных предметов (машина, орёл, сердце) и казахских \
предметов (домбра, юрта, тулпар) не нужно. Найденная справка (или описание и картинка от пользователя, \
если интернета нет) сама попадёт в бриф: после неё вызывай write_brief. Если пользователь выбрал «Делай \
как понимаешь», делай по тому, что знаешь.
1. write_brief — перед первой картинкой фигуры и когда задача меняется. Промпт-инженер пишет точное \
английское описание, стиль картинки, способ 3D и список проверок.
1а. Если в брифе есть template (у домбры, шанырака, юрты, Байтерека есть точный чертёж), сразу вызови \
build_exact: он строит нужные детали и формы за секунду, без картинок. Если пользователь просит что-то \
изменить (длиннее гриф, шире корпус), вызови build_exact снова с adjust (имена и пределы параметров смотри в \
его ответе). Нарисованный нейросетью вариант (шаги 2-4) делай, только если пользователь просит рисунок.
2. generate_images — по брифу, обычно 2 варианта; в subject пиши описание из брифа (стиль и вид берутся \
из брифа сами). Программа сама считает крупные детали на каждой картинке и пишет результат: «НЕ прошла» \
значит, что деталей неверное число. Такую картинку не предлагай: исправь subject (явно опиши правильную \
деталь) и нарисуй заново; больше двух повторов подряд не делай — тогда скажи, что не получается, и спроси \
пользователя. Остальное (форму, пропорции, лишние предметы) проверь по картинке сам.
3. Покажи удачные варианты и спроси инструментом ask_user, какой брать; варианты ответа — номера удачных \
картинок, «Ещё варианты» и «Изменить». Если удачный вариант один и пользователь просил сразу 3D — делай 3D.
4. make_3d — по выбранной картинке (для листа видов — по листу); способ и симметрию берёт бриф. \
Посмотри на превью модели: если форма явно неверна, скажи и предложи, как исправить.
4а. Если пользователь прислал 2-4 фото одного предмета с разных сторон (в сообщении у фото указан вид), сразу вызови make_3d: фото склеятся в лист и пойдут в 3D, картинки рисовать не нужно. Переключатель «Точность» чата (одна картинка или лист из 4 видов) уже учтён в брифе: не переспрашивай про это. \
В photos передавай только такие фото пользователя; нарисованную картинку туда не подставляй: одна картинка — \
это один вид, другие стороны из неё не получить.
5. Предложи отправить модель в Blender (send_to_blender) или отправь сразу, если пользователь об этом просил.

Цвета модели (пользователь может называть их «текстурой»)
- Цвета переносятся на модель с картинок. С одной картинки видна только одна сторона: бока и спина \
раскрашены приблизительно, без деталей. Точные цвета и форма со всех сторон — только по листу из 4 видов.
- Если пользователь недоволен цветами или формой, не вызывай make_3d по той же картинке: выйдет то же самое. \
Спроси через ask_user, что сделать, с вариантами «Лист из 4 видов (точнее со всех сторон)», «Другой вариант \
картинки», «Оставить как есть». Для листа вызови write_brief, потом generate_images и make_3d по листу.
- Не пиши, что что-то исправлено или стало лучше, если этого не видно в новом результате инструмента.

Если инструмент вернул ошибку, скажи о ней простыми словами и предложи, что сделать. Не выдумывай \
результатов, которых нет.
"""
# the small local model tends to write a call as text and to approve pictures too easily
LOCAL_NOTE = """
Важно: инструменты вызывай только как функции, никогда не пиши их названия в тексте ответа. Проверяя \
картинку, сначала перечисли, что на ней видно (сколько струн, колков, ног, крыльев), и только потом \
сравнивай с проверками брифа. Не пиши «отправляю в Blender» или «отправил», пока не вызвал send_to_blender; \
если пользователь об отправке не просил, скажи, что модель можно отправить кнопкой «В Blender».
"""


def _closing_results(message: dict, text: str) -> dict | None:
    """Error results for every tool call of ``message`` (keeps the history valid)."""
    uses = tool_uses(message)
    if not uses:
        return None
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": u["id"], "is_error": True, "content": [{"type": "text", "text": text}],
         "ui": {"kind": "error", "text": text}} for u in uses]}


def display(message: dict) -> dict:
    """A stored message as the Studio shows it (without the provider's raw blocks)."""
    return {k: v for k, v in message.items() if k != "raw"}


VIEW_RU = T.VIEW_RU


class ChatService:
    """Runs chat turns (one at a time per chat) and fans their events out to listeners."""

    def __init__(self, settings, store, config, providers: Providers, manager):
        self.settings, self.store, self.config = settings, store, config
        self.providers, self.manager = providers, manager
        self.image_dir: Path = providers.image_dir
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._live: dict[str, dict] = {}

    # -- listeners ----------------------------------------------------------------------
    def subscribe(self, chat_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subs.setdefault(chat_id, set()).add(q)
        return q

    def unsubscribe(self, chat_id: str, q: asyncio.Queue) -> None:
        self._subs.get(chat_id, set()).discard(q)

    async def publish(self, chat_id: str, event: dict) -> None:
        event = {**event, "chat_id": chat_id, "t": time.time()}
        live = self._live.get(chat_id)
        if live is not None:
            self._track(live, event)
        listeners = list(self._subs.get(chat_id, ())) + list(self._subs.get("*", ()))
        for q in listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a stuck listener must not stop the turn
                pass

    def _track(self, live: dict, event: dict) -> None:
        """What a listener that connects mid-turn needs to redraw the turn so far."""
        kind = event["type"]
        if kind == "assistant_start":
            live.update(text="", notes="", tool=None)
        elif kind == "text":
            live["text"] += event["delta"]
        elif kind == "note":
            live["notes"] += event["delta"]
        elif kind in ("tool_call", "tool_running", "tool_progress"):
            live["tool"] = {k: v for k, v in event.items() if k not in ("chat_id", "t")}
        elif kind in ("tool_result", "assistant_message"):
            live["tool"] = None
            if kind == "assistant_message":
                live.update(text="", notes="")

    def live(self, chat_id: str) -> dict | None:
        return self._live.get(chat_id) if self.busy(chat_id) else None

    def busy(self, chat_id: str) -> bool:
        task = self._tasks.get(chat_id)
        return task is not None and not task.done()

    # -- turns --------------------------------------------------------------------------
    def user_message(self, text: str, uploads: list[tuple[Image.Image, str]] = (), answers=None, views=()) -> dict:
        """A user message in the neutral format; uploads are (picture, file name in outputs). ``views``: the side
        of the object each upload shows (the user chose it), "" when not said."""
        content = []
        for i, (img, name) in enumerate(uploads):
            view = views[i] if i < len(views) and views[i] in VIEW_RU else ""
            block = {"type": "image", "file": llm_images.add(self.image_dir, img), "source": name}
            if view:
                block["view"] = view
            content.append(block)
            content.append({"type": "text", "text": f"(картинка пользователя: файл {name}" + (f", вид: {VIEW_RU[view]})" if view else ")")})
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        return {"role": "user", "content": content, "answers": answers or None}

    def send(self, chat_id: str, message: dict) -> None:
        """Store the user's message and start a turn (raises RuntimeError when one is running)."""
        if self.busy(chat_id):
            raise RuntimeError("Ассистент ещё отвечает — дождитесь или нажмите «Стоп»")
        chat = self.store.chat(chat_id)
        pending = chat["state"].get("pending")
        if pending:
            message = self._answer(chat_id, pending, message)
            self.store.set_state(chat_id, pending=None)
        mid = self.store.add_message(chat_id, message)
        self._live[chat_id] = {"text": "", "notes": "", "tool": None, "started": time.time()}
        self._tasks[chat_id] = asyncio.get_running_loop().create_task(self._turn(chat_id, {**message, "id": mid}))

    def _answer(self, chat_id: str, pending: dict, message: dict) -> dict:
        """The user's reply to ``ask_user`` (or to a tool's question) becomes that call's result (plus earlier results)."""
        lines = []
        for q, a in zip(pending.get("questions", []), message.get("answers") or []):
            if a:
                lines.append(f"«{q['question']}» — {', '.join(a) if isinstance(a, list) else a}")
        free = " ".join(b["text"] for b in message["content"] if b["type"] == "text").strip()
        said = " ".join(b["text"] for b in message["content"]
                        if b["type"] == "text" and not b["text"].startswith("(картинка пользователя")).strip()
        pictures = [b for b in message["content"] if b["type"] == "image"]
        text = ("Ответы: " + "; ".join(lines) + ("\n" if lines and free else "") if lines else "") + free
        text = text or "(пользователь ничего не ответил)"
        ui = {"kind": "answer", "answers": message.get("answers"), "text": said}
        if pending.get("reference"):  # how the thing looks, in the user's words or picture: the brief uses it
            if len(said) >= 3 or pictures:
                self._keep_reference(chat_id, pending["reference"], said, pictures)
                text += "\n(Это сохранено как справка о том, как выглядит фигура: само попадёт в бриф. Вызывай write_brief.)"
            ui["lookup"] = {**pending.get("ui", {}), "questions": pending.get("questions", [])}
        if pending.get("intro"):
            text = f"{pending['intro']}\n{text}"
        result = {"type": "tool_result", "tool_use_id": pending["tool_use_id"], "is_error": False,
                  "content": [{"type": "text", "text": text}] + pictures, "ui": ui}
        return {"role": "user", "content": [*pending.get("results", []), result], "answers": message.get("answers")}

    def _keep_reference(self, chat_id: str, name: str, said: str, pictures: list[dict]) -> None:
        picture = pictures[0] if pictures else {}
        files = {"image": picture["source"]} if picture.get("source") else {}
        self.store.add_asset(chat_id, "reference", name, files,
                             {"name": name, "title": name, "url": "", "text": said, "text_en": "", "source": "user",
                              "picture": picture.get("file", "")})

    async def stop(self, chat_id: str) -> bool:
        task = self._tasks.get(chat_id)
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return True

    async def close(self) -> None:
        for chat_id in list(self._tasks):
            await self.stop(chat_id)

    async def _turn(self, chat_id: str, message: dict) -> None:
        async def publish(event: dict) -> None:
            if event["type"] == "notice" and event.get("level") != "info":
                self.store.add_message(chat_id, {"role": "notice", "level": event["level"], "text": event["text"]})
            await self.publish(chat_id, event)

        await publish({"type": "user_message", "message": display(message)})
        provider = self.providers.pick()
        await publish({"type": "provider", "name": provider.name, "title": provider.title,
                       "cloud": provider is self.providers.claude})
        status = "done"
        last_assistant = None
        nudged, nudge = False, None
        try:
            for _step in range(MAX_STEPS):
                history = self.store.messages(chat_id) + ([nudge] if nudge else [])  # the nudge is not stored
                nudge = None
                await publish({"type": "assistant_start"})

                async def on_event(ev):
                    if ev.kind == "text":
                        await publish({"type": "text", "delta": ev.text, "index": ev.data.get("index", 0)})
                    elif ev.kind == "note":
                        await publish({"type": "note", "delta": ev.text, "index": ev.data.get("index", 0)})
                    elif ev.kind == "tool_start":
                        await publish({"type": "tool_call", **ev.data, "title": T.TITLES.get(ev.data.get("name"), "")})
                    elif ev.kind == "reset":
                        await publish({"type": "assistant_start"})

                system = SYSTEM + (LOCAL_NOTE if provider is self.providers.local else "")
                try:
                    reply = await provider.respond(system, history, T.TOOLS, EFFORT, on_event)
                except LLMError as e:
                    alt = self.providers.fallback(provider) if e.offline else None
                    if alt is None:
                        raise
                    await publish({"type": "notice", "level": "warn",
                                   "text": f"{e} Продолжаю на локальной модели."})
                    provider = alt
                    await publish({"type": "provider", "name": provider.name, "title": provider.title, "cloud": False})
                    continue
                if reply["stop"] == "refusal":
                    category = reply.get("stop_category")
                    await publish({"type": "notice", "level": "warn", "text": "Модель отказалась отвечать на это сообщение"
                                   + (f" (причина: {category})" if category else "") + ". Переформулируйте запрос."})
                    status = "refused"
                    break
                reply["id"] = self.store.add_message(chat_id, reply)
                last_assistant = reply
                await publish({"type": "assistant_message", "message": display(reply)})
                uses = tool_uses(reply)
                if not uses:
                    if reply["stop"] == "max_tokens":
                        await publish({"type": "notice", "level": "warn", "text": "Ответ оборвался: слишком длинный."})
                    said = " ".join(b["text"] for b in reply["content"] if b["type"] == "text")
                    if provider is self.providers.local and not nudged and PROMISE.search(said):
                        nudged, nudge = True, {"role": "user", "content": [{"type": "text", "text": NUDGE}]}
                        last_assistant = None
                        continue
                    break
                if reply["stop"] == "max_tokens":  # a cut-off tool input must not run
                    closing = _closing_results(reply, "Вызов оборвался на лимите длины ответа — повтори его короче")
                    self.store.add_message(chat_id, closing)
                    last_assistant = None
                    continue
                ctx = T.ToolContext(chat_id, self.settings, self.store, self.config, self.manager, provider,
                                    self.image_dir, publish)
                results, question = [], None  # question: (call, questions, what else the answer needs)
                for use in uses:
                    if use["name"] == "ask_user" and question is None:
                        try:
                            ask = T.BY_NAME["ask_user"].parse(use["input"])
                            question = (use, [q.model_dump() for q in ask.questions], {})
                            continue
                        except ValueError as e:
                            result = {"type": "tool_result", "tool_use_id": use["id"], "is_error": True,
                                      "content": [{"type": "text", "text": f"Неверные параметры ask_user: {e}"}]}
                    elif use["name"] == "ask_user":
                        result = {"type": "tool_result", "tool_use_id": use["id"], "is_error": True,
                                  "content": [{"type": "text", "text": "Задавай все вопросы одним вызовом ask_user"}]}
                    else:
                        result = await T.execute(ctx, use)
                        ask = result.pop("ask", None)
                        if ask is not None and question is None:  # the tool asks the user itself
                            if ask["notice"]:
                                await publish({"type": "notice", "level": "warn", "text": ask["notice"]})
                            await publish({"type": "tool_result", "tool_use_id": use["id"], "ok": True, "ui": result.get("ui"),
                                           "text": ""})
                            question = (use, ask["questions"], {k: ask[k] for k in ("intro", "reference", "ui")})
                            continue
                    results.append(result)
                    await publish({"type": "tool_result", "tool_use_id": use["id"], "ok": not result["is_error"],
                                   "ui": result.get("ui"), "text": "\n".join(c.get("text", "") for c in result["content"]
                                                                              if c["type"] == "text")})
                if question is not None:
                    use, questions, extra = question
                    self.store.set_state(chat_id, pending={"tool_use_id": use["id"], "questions": questions,
                                                           "results": results, **extra})
                    await publish({"type": "choices", "tool_use_id": use["id"], "questions": questions})
                    last_assistant = None
                    status = "waiting"
                    break
                self.store.add_message(chat_id, {"role": "user", "content": results})
                last_assistant = None
            else:
                await publish({"type": "notice", "level": "warn", "text": "Слишком много шагов подряд — остановился."})
        except asyncio.CancelledError:
            status = "stopped"
            self._close_open_calls(chat_id, last_assistant, "Остановлено пользователем")
            await publish({"type": "notice", "level": "info", "text": "Остановлено."})
        except LLMError as e:
            status = "error"
            self._close_open_calls(chat_id, last_assistant, str(e))
            await publish({"type": "notice", "level": "error", "text": str(e)})
        except Exception as e:  # noqa: BLE001 - a turn failure must reach the chat, not just the log
            log.exception("chat turn failed")
            status = "error"
            self._close_open_calls(chat_id, last_assistant, "Внутренняя ошибка")
            await publish({"type": "notice", "level": "error", "text": f"Внутренняя ошибка: {e}"})
        finally:
            self._live.pop(chat_id, None)
            await publish({"type": "turn_end", "status": status})

    def _close_open_calls(self, chat_id: str, last_assistant: dict | None, text: str) -> None:
        if last_assistant is None:
            return
        history = self.store.messages(chat_id)
        if history and history[-1].get("id") == last_assistant.get("id"):
            closing = _closing_results(last_assistant, text)
            if closing:
                self.store.add_message(chat_id, closing)
