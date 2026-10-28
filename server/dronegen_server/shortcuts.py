"""The «DroneGen Studio» shortcut: one click starts the server without a console window and opens the chat.

``python -m dronegen_server --make-shortcuts`` (install.py runs it too) puts it on the desktop and in the Start menu
(where Windows search finds it, and where it can be pinned to the taskbar).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

NAME = "DroneGen Studio"


def make_icon(path: Path) -> Path:
    """The logo of the Studio (a rounded square with five drones) as a multi-size .ico."""
    from PIL import Image, ImageDraw

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gradient = Image.new("RGBA", (size, size))
    px = gradient.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size - 2)
            px[x, y] = (int(79 + (138 - 79) * t), int(142 + (92 - 142) * t), int(247 + (246 - 247) * t), 255)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 4, fill=255)
    img.paste(gradient, (0, 0), mask)
    draw = ImageDraw.Draw(img)
    r = size * 0.069
    for cx, cy in ((16, 8), (9, 15), (23, 15), (12, 23), (20, 23)):  # the same dots as the page icon (32 px grid)
        x, y = cx * size / 32, cy * size / 32
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    return path


_PS = r"""
param($home_, $py, $icon, $name)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$shell = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath('Desktop'), (Join-Path ([Environment]::GetFolderPath('Programs')) ''))
foreach ($dir in $targets) {
  if (-not $dir -or -not (Test-Path $dir)) { continue }
  $lnk = $shell.CreateShortcut((Join-Path $dir ($name + '.lnk')))
  $lnk.TargetPath = $py
  $lnk.Arguments = '-m dronegen_server --home "' + $home_ + '" --app'
  $lnk.WorkingDirectory = $home_
  $lnk.IconLocation = $icon
  $lnk.Description = 'Text to image to 3D for drone shows (local)'
  $lnk.Save()
  Write-Output ('ярлык: ' + (Join-Path $dir ($name + '.lnk')))
}
"""


def create_shortcuts(home: Path) -> list[str]:
    """Create the shortcuts; returns a line per result (Windows only)."""
    if sys.platform != "win32":
        return ["ярлыки поддерживаются только в Windows: запускайте start_server.sh --open-studio"]
    home = home.resolve()
    pythonw = home / "venv" / "Scripts" / "pythonw.exe"
    if not pythonw.is_file():
        return [f"не найден {pythonw}"]
    icon = make_icon(home / "dronegen.ico")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "shortcut.ps1"
        script.write_text(_PS, encoding="utf-8-sig")  # the BOM makes Windows PowerShell 5 read it as UTF-8
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script),
             str(home), str(pythonw), str(icon), NAME],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if out.returncode != 0 or not lines:
        lines.append("не удалось создать ярлык: " + (out.stderr.strip() or "PowerShell ничего не вернул"))
    return lines
