"""App mode: the server follows the life of the Studio window."""

import threading
import time

from dronegen_server import launcher


def test_wait_returns_when_window_closes(monkeypatch, tmp_path):
    alive = {"n": 0}
    monkeypatch.setattr(launcher, "_window_processes", lambda profile: ["p"] if alive["n"] else [])
    done = []

    def run():
        done.append(launcher.wait_window_closed(tmp_path, appear_s=5, poll_s=0.05))

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.2)
    alive["n"] = 1  # the window appears
    time.sleep(0.2)
    assert not done  # still open
    alive["n"] = 0  # and is closed
    t.join(3)
    assert done == [True]


def test_wait_gives_up_when_no_window_appears(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_window_processes", lambda profile: [])
    assert launcher.wait_window_closed(tmp_path, appear_s=0.3) is False


def test_shortcut_icon_is_a_valid_ico(tmp_path):
    from PIL import Image

    from dronegen_server.shortcuts import make_icon

    ico = Image.open(make_icon(tmp_path / "x.ico"))
    assert ico.format == "ICO" and 256 in {s[0] for s in ico.info.get("sizes", {(256, 256)})}
