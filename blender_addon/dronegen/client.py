"""HTTP client for the DroneGen server. Standard library only, no bpy: safe in threads."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ServerError(Exception):
    def __init__(self, message: str, status: int | None = None, offline: bool = False):
        super().__init__(message)
        self.status = status
        self.offline = offline  # nothing listens at the address: the server is not running


def is_local(host: str | None) -> bool:
    return bool(host) and (host in ("localhost", "::1") or host.startswith("127."))


class Client:
    def __init__(self, base_url: str, token: str = "", timeout: float = 15.0):
        self.base = (base_url or "").strip().rstrip("/")
        parts = urllib.parse.urlsplit(self.base)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ServerError(f"Неверный адрес сервера: «{base_url}». Пример: http://127.0.0.1:8188")
        self.host = parts.hostname
        self.port = parts.port
        self.token = token
        self.timeout = timeout
        # a system proxy must never see requests to this computer
        handlers = [urllib.request.ProxyHandler({})] if is_local(self.host) else []
        self._opener = urllib.request.build_opener(*handlers)

    @property
    def local(self) -> bool:
        return is_local(self.host)

    # -- low level ------------------------------------------------------------------------
    def _open(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None, timeout=None):
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            return self._opener.open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raise ServerError(_detail(e), e.code) from None
        except (urllib.error.URLError, OSError) as e:
            raise ServerError(_connection_message(self.base, e), offline=not _timed_out(e)) from None

    def _json(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None, timeout=None):
        with self._open(method, path, data, headers, timeout) as resp:
            body = resp.read()
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError:
            raise ServerError(f"Ответ {self.base} не похож на сервер DroneGen (не JSON)") from None

    # -- API ------------------------------------------------------------------------------
    def health(self) -> dict:
        data = self._json("GET", "/health", timeout=5)
        if data.get("name") != "dronegen":
            raise ServerError(f"По адресу {self.base} отвечает не сервер DroneGen")
        return data

    def txt2img(self, params: dict) -> dict:
        body = json.dumps(params).encode("utf-8")
        return self._json("POST", "/txt2img", body, {"Content-Type": "application/json"})

    def img2mesh(self, image_path: str | Path, params: dict) -> dict:
        fields = {k: _form_value(v) for k, v in params.items() if v is not None}
        body, ctype = _multipart(fields, "image", Path(image_path))
        return self._json("POST", "/img2mesh", body, {"Content-Type": ctype}, timeout=max(self.timeout, 60))

    def rig(self, mesh_npz: bytes, params: dict) -> dict:
        """A skeleton for a mesh (NPZ with vertices and faces in the object's space): a job id."""
        fields = {k: _form_value(v) for k, v in params.items() if v is not None}
        body, ctype = _multipart_bytes(fields, "mesh", "mesh.npz", mesh_npz, "application/octet-stream")
        return self._json("POST", "/rig", body, {"Content-Type": ctype}, timeout=max(self.timeout, 60))

    def rig_plans(self) -> dict:
        return self._json("GET", "/rig/plans", timeout=5)

    def motions(self) -> dict:
        return self._json("GET", "/motions", timeout=5)

    def bake(self, skeleton: dict, motion: str, params: dict | None = None, samples: int = 24) -> dict:
        """Keys of a motion for a skeleton (answered at once, not queued)."""
        body = json.dumps({"skeleton": skeleton, "motion": motion, "params": params or {}, "samples": samples})
        return self._json("POST", "/motion/bake", body.encode("utf-8"), {"Content-Type": "application/json"})

    def effect(self, request: dict) -> dict:
        """Places of an effect's drones and their waiting places in the reserve (answered at once)."""
        body = json.dumps(request).encode("utf-8")
        return self._json("POST", "/effects/build", body, {"Content-Type": "application/json"}, timeout=max(self.timeout, 30))

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/jobs/{urllib.parse.quote(job_id)}")

    def cancel(self, job_id: str) -> dict:
        return self._json("POST", f"/jobs/{urllib.parse.quote(job_id)}/cancel", b"")

    # -- the Studio's inbox: models sent with «В Blender» ----------------------------------
    def inbox(self) -> list:
        return self._json("GET", "/blender/inbox", timeout=5)

    def inbox_taken(self, item_id: str) -> dict:
        return self._json("POST", f"/blender/inbox/{urllib.parse.quote(item_id)}/taken", b"")

    def inbox_done(self, item_id: str, obj: str = "", error: str = "") -> dict:
        body = json.dumps({"object": obj, "error": error}).encode("utf-8")
        return self._json("POST", f"/blender/inbox/{urllib.parse.quote(item_id)}/done", body,
                          {"Content-Type": "application/json"})

    def studio_url(self) -> str:
        """Address of the Studio web app (the token, if any, rides along for its event streams)."""
        url = self.base + "/studio/"
        return url + "?token=" + urllib.parse.quote(self.token) if self.token else url

    def unload(self) -> dict:
        return self._json("POST", "/unload", b"")

    def shutdown(self) -> dict:
        return self._json("POST", "/shutdown", b"")

    def download(self, name: str, dest: Path) -> Path:
        """Stream a result file into ``dest`` (written atomically)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with self._open("GET", f"/files/{urllib.parse.quote(name)}", timeout=max(self.timeout, 60)) as resp, open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(part, dest)
        return dest


def _form_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _multipart(fields: dict, file_field: str, path: Path) -> tuple[bytes, str]:
    ext = path.suffix.lower() or ".png"
    ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "application/octet-stream")
    return _multipart_bytes(fields, file_field, f"image{ext}", path.read_bytes(), ctype)


def _multipart_bytes(fields: dict, file_field: str, filename: str, data: bytes, ctype: str) -> tuple[bytes, str]:
    boundary = "dronegen-" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode()
        out += value.encode("utf-8") + b"\r\n"
    out += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _detail(e: urllib.error.HTTPError) -> str:
    try:
        data = json.loads(e.read().decode("utf-8"))
        detail = data.get("detail")
    except Exception:  # noqa: BLE001 - not JSON
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(str(x.get("msg", x)) if isinstance(x, dict) else str(x) for x in detail)
    if e.code == 401:
        return detail or "Сервер требует токен доступа (настройки аддона → Токен)"
    return detail or f"Сервер вернул ошибку HTTP {e.code}"


def _timed_out(e: Exception) -> bool:
    reason = getattr(e, "reason", e)
    return isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason).lower()


def _connection_message(base: str, e: Exception) -> str:
    if _timed_out(e):
        return f"Сервер {base} не ответил вовремя: возможно, он занят загрузкой модели. Попробуйте ещё раз."
    return (
        f"Нет связи с сервером {base}. Запустите сервер (кнопка «Запустить сервер» или start_server.bat) "
        "и проверьте адрес в настройках аддона."
    )
