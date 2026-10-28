"""Studio data: projects, chats, messages, assets and the Blender inbox (SQLite), plus settings.

Everything lives in ``<home>/studio``: ``studio.db``, ``settings.json`` and ``secrets.json``
(the Claude API key and the Sketchfab token; ``runtime/`` is not in git, and the HTTP API never
returns a secret, only whether it is set).

A project is a show («Наурыз-27»), a chat is one figure or idea in it. A message keeps the
neutral chat format of ``llm.base`` (with the provider's raw blocks). An asset is something a
chat made - a reference (how a named character looks), a brief, pictures, cards, a mesh - with its
parent, so a mesh knows its picture and the picture its prompt.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, pinned INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, kind TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '', files TEXT NOT NULL DEFAULT '{}', data TEXT NOT NULL DEFAULT '{}',
    parent TEXT, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    id TEXT PRIMARY KEY, chat_id TEXT, asset_id TEXT, file TEXT NOT NULL, name TEXT NOT NULL,
    status TEXT NOT NULL, object TEXT, created REAL NOT NULL, updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
CREATE INDEX IF NOT EXISTS chats_project ON chats(project_id, updated);
CREATE INDEX IF NOT EXISTS assets_chat ON assets(chat_id, created);
"""

DEFAULTS = {
    "provider": "local",  # local (nothing leaves the computer) | claude (the cloud, opt-in)
    "claude_model": "claude-opus-5-5",
    "image_backend": "zimage_turbo",
    "variants": 2,  # pictures per request
    "web_lookup": True,  # look up unknown characters on Wikipedia (only their name goes out); False: ask the user
}
SECRETS = ("anthropic_api_key", "sketchfab_token")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class NotFound(LookupError):
    pass


class Store:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.folder / "studio.db", check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # SQLite's LIKE and lower() fold only ASCII: search Cyrillic with Python's lower()
        self._db.create_function("ulower", 1, lambda s: s.lower() if isinstance(s, str) else s, deterministic=True)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            return cur.lastrowid if cur.lastrowid is not None else cur.rowcount

    # -- projects -----------------------------------------------------------------------
    def projects(self) -> list[dict]:
        rows = self._q(
            "SELECT p.*, (SELECT COUNT(*) FROM chats c WHERE c.project_id = p.id) AS chats "
            "FROM projects p ORDER BY p.pinned DESC, p.updated DESC"
        )
        return [dict(r) for r in rows]

    def project(self, project_id: str) -> dict:
        rows = self._q("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not rows:
            raise NotFound("Проект не найден")
        return dict(rows[0])

    def add_project(self, title: str) -> dict:
        now = time.time()
        pid = new_id()
        self._x("INSERT INTO projects (id, title, created, updated) VALUES (?, ?, ?, ?)", (pid, title.strip() or "Проект", now, now))
        return self.project(pid)

    def update_project(self, project_id: str, title: str | None = None, pinned: bool | None = None) -> dict:
        self.project(project_id)
        if title is not None:
            self._x("UPDATE projects SET title = ?, updated = ? WHERE id = ?", (title.strip() or "Проект", time.time(), project_id))
        if pinned is not None:
            self._x("UPDATE projects SET pinned = ? WHERE id = ?", (int(pinned), project_id))
        return self.project(project_id)

    def delete_project(self, project_id: str) -> None:
        self.project(project_id)
        self._x("DELETE FROM projects WHERE id = ?", (project_id,))

    # -- chats --------------------------------------------------------------------------
    def chats(self, project_id: str | None = None, query: str = "") -> list[dict]:
        sql = "SELECT c.*, p.title AS project_title FROM chats c JOIN projects p ON p.id = c.project_id"
        where, args = [], []
        if project_id:
            where.append("c.project_id = ?")
            args.append(project_id)
        if query:
            where.append("(ulower(c.title) LIKE ? OR EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = c.id "
                         "AND ulower(m.body) LIKE ?))")
            args += [f"%{query.lower()}%"] * 2
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY c.pinned DESC, c.updated DESC"
        return [self._chat_row(r) for r in self._q(sql, args)]

    def _chat_row(self, row) -> dict:
        d = dict(row)
        d["state"] = json.loads(d.get("state") or "{}")
        return d

    def chat(self, chat_id: str) -> dict:
        rows = self._q("SELECT c.*, p.title AS project_title FROM chats c JOIN projects p ON p.id = c.project_id WHERE c.id = ?",
                       (chat_id,))
        if not rows:
            raise NotFound("Чат не найден")
        return self._chat_row(rows[0])

    def add_chat(self, project_id: str, title: str = "Новый чат") -> dict:
        self.project(project_id)
        now = time.time()
        cid = new_id()
        self._x("INSERT INTO chats (id, project_id, title, created, updated) VALUES (?, ?, ?, ?, ?)",
                (cid, project_id, title.strip() or "Новый чат", now, now))
        self._x("UPDATE projects SET updated = ? WHERE id = ?", (now, project_id))
        return self.chat(cid)

    def update_chat(self, chat_id: str, title: str | None = None, pinned: bool | None = None,
                    project_id: str | None = None) -> dict:
        self.chat(chat_id)
        if title is not None:
            self._x("UPDATE chats SET title = ? WHERE id = ?", (title.strip() or "Чат", chat_id))
        if pinned is not None:
            self._x("UPDATE chats SET pinned = ? WHERE id = ?", (int(pinned), chat_id))
        if project_id is not None:
            self.project(project_id)
            self._x("UPDATE chats SET project_id = ? WHERE id = ?", (project_id, chat_id))
        return self.chat(chat_id)

    def set_state(self, chat_id: str, **changes) -> dict:
        """Update keys of the chat's state (None removes a key)."""
        with self._lock:
            state = self.chat(chat_id)["state"]
            for k, v in changes.items():
                if v is None:
                    state.pop(k, None)
                else:
                    state[k] = v
            self._x("UPDATE chats SET state = ? WHERE id = ?", (json.dumps(state, ensure_ascii=False), chat_id))
            return state

    def touch(self, chat_id: str) -> None:
        now = time.time()
        self._x("UPDATE chats SET updated = ? WHERE id = ?", (now, chat_id))
        self._x("UPDATE projects SET updated = ? WHERE id = (SELECT project_id FROM chats WHERE id = ?)", (now, chat_id))

    def delete_chat(self, chat_id: str) -> None:
        self.chat(chat_id)
        self._x("DELETE FROM chats WHERE id = ?", (chat_id,))

    # -- messages -----------------------------------------------------------------------
    def add_message(self, chat_id: str, message: dict) -> int:
        mid = self._x("INSERT INTO messages (chat_id, role, body, created) VALUES (?, ?, ?, ?)",
                      (chat_id, message["role"], json.dumps(message, ensure_ascii=False), time.time()))
        self.touch(chat_id)
        return mid

    def messages(self, chat_id: str) -> list[dict]:
        """The whole history, oldest first: each message with its ``id`` and ``created``."""
        rows = self._q("SELECT id, body, created FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))
        out = []
        for r in rows:
            m = json.loads(r["body"])
            m["id"], m["created"] = r["id"], r["created"]
            out.append(m)
        return out

    # -- assets -------------------------------------------------------------------------
    def add_asset(self, chat_id: str, kind: str, title: str = "", files: dict | None = None, data: dict | None = None,
                  parent: str | None = None) -> dict:
        aid = new_id()
        self._x("INSERT INTO assets (id, chat_id, kind, title, files, data, parent, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (aid, chat_id, kind, title, json.dumps(files or {}, ensure_ascii=False),
                 json.dumps(data or {}, ensure_ascii=False), parent, time.time()))
        return self.asset(aid)

    def _asset_row(self, row) -> dict:
        d = dict(row)
        d["files"], d["data"] = json.loads(d["files"]), json.loads(d["data"])
        return d

    def asset(self, asset_id: str) -> dict:
        rows = self._q("SELECT * FROM assets WHERE id = ?", (asset_id,))
        if not rows:
            raise NotFound("Ассет не найден")
        return self._asset_row(rows[0])

    def assets(self, chat_id: str, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM assets WHERE chat_id = ?", [chat_id]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        return [self._asset_row(r) for r in self._q(sql + " ORDER BY created", args)]

    def update_asset(self, asset_id: str, data: dict | None = None, files: dict | None = None) -> dict:
        self.asset(asset_id)
        if data is not None:
            self._x("UPDATE assets SET data = ? WHERE id = ?", (json.dumps(data, ensure_ascii=False), asset_id))
        if files is not None:
            self._x("UPDATE assets SET files = ? WHERE id = ?", (json.dumps(files, ensure_ascii=False), asset_id))
        return self.asset(asset_id)

    def asset_by_file(self, chat_id: str, name: str) -> dict | None:
        """The asset of this chat that has the output file ``name``."""
        for a in reversed(self.assets(chat_id)):
            if name in a["files"].values():
                return a
        return None

    # -- the Blender inbox --------------------------------------------------------------
    def add_inbox(self, file: str, name: str, chat_id: str | None = None, asset_id: str | None = None) -> dict:
        iid, now = new_id(), time.time()
        self._x("INSERT INTO inbox (id, chat_id, asset_id, file, name, status, created, updated) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                (iid, chat_id, asset_id, file, name, now, now))
        return self.inbox_item(iid)

    def inbox_item(self, item_id: str) -> dict:
        rows = self._q("SELECT * FROM inbox WHERE id = ?", (item_id,))
        if not rows:
            raise NotFound("Нет такой посылки для Blender")
        return dict(rows[0])

    def inbox(self, status: str | None = "queued") -> list[dict]:
        if status:
            return [dict(r) for r in self._q("SELECT * FROM inbox WHERE status = ? ORDER BY created", (status,))]
        return [dict(r) for r in self._q("SELECT * FROM inbox ORDER BY created DESC LIMIT 50")]

    def update_inbox(self, item_id: str, status: str, obj: str | None = None) -> dict:
        self.inbox_item(item_id)
        self._x("UPDATE inbox SET status = ?, object = COALESCE(?, object), updated = ? WHERE id = ?",
                (status, obj, time.time(), item_id))
        return self.inbox_item(item_id)


class StudioConfig:
    """``settings.json`` (provider, model...) and ``secrets.json`` (keys) in the studio folder."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read(self, name: str) -> dict:
        path = self.folder / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, name: str, data: dict) -> None:
        path = self.folder / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def get(self, key: str):
        return self._read("settings.json").get(key, DEFAULTS.get(key))

    def all(self) -> dict:
        return {**DEFAULTS, **self._read("settings.json")}

    def update(self, **values) -> dict:
        with self._lock:
            data = self._read("settings.json")
            data.update({k: v for k, v in values.items() if k in DEFAULTS and v is not None})
            self._write("settings.json", data)
            return {**DEFAULTS, **data}

    def secret(self, key: str) -> str:
        return str(self._read("secrets.json").get(key) or "")

    def set_secret(self, key: str, value: str) -> None:
        if key not in SECRETS:
            raise KeyError(key)
        with self._lock:
            data = self._read("secrets.json")
            if value:
                data[key] = value.strip()
            else:
                data.pop(key, None)
            self._write("secrets.json", data)

    def secrets_state(self) -> dict:
        """Which secrets are set, with only the last 4 characters shown."""
        data = self._read("secrets.json")
        return {k: ({"set": True, "hint": "…" + str(data[k])[-4:]} if data.get(k) else {"set": False}) for k in SECRETS}
