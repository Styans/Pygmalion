"""GPU and RAM checks, peak measurements and clean-up.

PyTorch is imported only when it is installed, so the light profile (no torch) works too.
On Windows the limit that matters for loading big models is the commit limit (RAM + page
file), not free RAM, so it is checked separately.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time

from .errors import RAM_HINT, VRAM_HINT, OutOfMemory

GB = 1024**3


def torch_or_none():
    try:
        import torch
    except Exception:  # noqa: BLE001 - not installed / broken install
        return None
    return torch


def cuda_ready() -> bool:
    torch = torch_or_none()
    return bool(torch is not None and torch.cuda.is_available())


def pick_device(setting: str = "auto") -> str:
    if setting == "cpu":
        return "cpu"
    if cuda_ready():
        return "cuda"
    if setting == "cuda":
        raise OutOfMemory("Видеокарта NVIDIA (CUDA) недоступна, а в настройках сервера выбрано --device cuda")
    return "cpu"


def gpu_info() -> dict:
    torch = torch_or_none()
    if torch is None:
        return {"available": False, "reason": "PyTorch не установлен (лёгкий профиль: только силуэт и тест)"}
    info = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        info.update(available=False, reason="CUDA недоступна: нет видеокарты NVIDIA или установлен PyTorch без CUDA")
        return info
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    info.update(
        available=True,
        name=props.name,
        capability=f"{props.major}.{props.minor}",
        total_gb=round(total / GB, 2),
        free_gb=round(free / GB, 2),
        used_by_server_gb=round(torch.cuda.memory_reserved() / GB, 2),
    )
    return info


def vram_free_gb() -> float:
    torch = torch_or_none()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / GB


def _windows_commit() -> tuple[float, float]:
    """(commit limit, available commit) in GB via GlobalMemoryStatusEx."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullTotalPageFile / GB, st.ullAvailPageFile / GB


def pagefile_auto() -> bool:
    """Windows: is the page file "system managed" (it grows when the commit limit is near)?"""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        key = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            files, _ = winreg.QueryValueEx(k, "PagingFiles")
    except OSError:
        return False
    entries = files if isinstance(files, list) else [files]
    return any(e.startswith("?:") or e.strip().endswith(" 0 0") for e in entries if e)


def system_memory() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    out = {"total_gb": round(vm.total / GB, 1), "available_gb": round(vm.available / GB, 1)}
    if sys.platform == "win32":
        limit, avail = _windows_commit()
        out.update(commit_limit_gb=round(limit, 1), commit_available_gb=round(avail, 1), pagefile_auto=pagefile_auto())
    return out


def available_ram_gb() -> float:
    """What a new allocation can still get.

    Windows counts RAM + page file (the commit limit). A system-managed page file grows on
    demand, so then free physical RAM is the better estimate; a fixed one is a hard limit.
    """
    import psutil

    ram = psutil.virtual_memory().available / GB
    if sys.platform == "win32":
        commit = _windows_commit()[1]
        return max(commit, ram) if pagefile_auto() else commit
    return ram


def require_ram(need_gb: float, what: str, warn=None) -> None:
    """Check memory before loading a model.

    Caches are emptied first. With a system-managed page file Windows can still make room
    (slower, by paging), so a moderate shortage is only a warning; a hard error is raised
    when less than half of what is needed is available, or the page file is fixed.
    """
    have = available_ram_gb()
    if have >= need_gb:
        return
    free_gpu()
    have = available_ram_gb()
    if have >= need_gb:
        return
    message = f"Для «{what}» нужно ~{need_gb:.0f} ГБ свободной памяти, доступно {have:.1f} ГБ. "
    if have >= need_gb * 0.5 and (sys.platform != "win32" or pagefile_auto()):
        if warn is not None:
            warn(message + "Работаю с файлом подкачки — может быть медленно. Закройте лишние программы.")
        return
    raise OutOfMemory(message + RAM_HINT)


def vram_hint(need_gb: float) -> str:
    return f"Свободно {vram_free_gb():.1f} ГБ видеопамяти, нужно ~{need_gb:.1f} ГБ. " + VRAM_HINT


def free_gpu() -> None:
    """Give memory back: the GPU cache and PyTorch's cache of pinned host memory.

    Moving models between the CPU and the GPU (offloading) goes through pinned RAM, and
    PyTorch keeps those blocks for reuse: without emptying that cache, a server that had
    Z-Image and SDXL loaded kept ~9 GB of RAM after unloading both (measured 2026-10-01).
    """
    gc.collect()
    torch = torch_or_none()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        empty_host = getattr(torch.cuda, "empty_host_cache", None) or getattr(torch._C, "_host_emptyCache", None)
        if empty_host is not None:
            empty_host()


def _process_memory_gb() -> float:
    import psutil

    mi = psutil.Process(os.getpid()).memory_info()
    return getattr(mi, "private", mi.rss) / GB


class PeakMonitor:
    """Measures one job: wall time, peak VRAM (PyTorch allocator and whole device) and process RAM."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.device_peak_gb = 0.0
        self.ram_peak_gb = 0.0
        self.baseline_device_gb = 0.0
        self.torch_peak_gb = None
        self.seconds = 0.0
        self._t0 = 0.0
        self._cuda = cuda_ready()

    def _device_used_gb(self) -> float:
        import torch

        free, total = torch.cuda.mem_get_info()
        return (total - free) / GB

    def _sample(self):
        while not self._stop.wait(self.interval):
            self._read()

    def _read(self):
        try:
            self.ram_peak_gb = max(self.ram_peak_gb, _process_memory_gb())
            if self._cuda:
                self.device_peak_gb = max(self.device_peak_gb, self._device_used_gb())
        except Exception:  # noqa: BLE001 - measuring must never break a job
            pass

    def __enter__(self):
        if self._cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
            self.baseline_device_gb = self._device_used_gb()
        self._read()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, name="peak-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._read()
        self.seconds = time.perf_counter() - self._t0
        if self._cuda:
            import torch

            self.torch_peak_gb = torch.cuda.max_memory_reserved() / GB
        return False

    def metrics(self) -> dict:
        m = {"seconds": round(self.seconds, 2), "ram_peak_gb": round(self.ram_peak_gb, 2)}
        if self._cuda:
            m.update(
                vram_peak_torch_gb=round(self.torch_peak_gb or 0.0, 2),
                vram_device_peak_gb=round(self.device_peak_gb, 2),
                vram_device_before_gb=round(self.baseline_device_gb, 2),
            )
        return m
