"""Errors with messages for the user (Russian) and translation of low-level failures."""

from __future__ import annotations


class UserError(Exception):
    """Bad input or a known, explainable problem; the message is shown in Blender as is."""


class BackendUnavailable(UserError):
    pass


class JobCancelled(Exception):
    pass


class OutOfMemory(UserError):
    """Not enough GPU or system memory even after the lighter fallbacks."""


def is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 - torch is optional
        pass
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text or "cudnn" in text)


def is_ram_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return (
        "defaultcpuallocator" in text
        or "not enough memory" in text
        or "winerror 1455" in text
        or "paging file is too small" in text
        or "файл подкачки" in text
    )


VRAM_HINT = (
    "Закройте программы, которые занимают видеопамять (EasyDiffusion, ComfyUI, игры, браузер с 3D), "
    "или выберите облегчённые настройки: меньше размер картинки или разрешение сетки, бэкенд «силуэт»."
)
RAM_HINT = (
    "Не хватает оперативной памяти: закройте лишние программы или увеличьте файл подкачки "
    "(Windows: Параметры → Система → О системе → Дополнительные параметры → Быстродействие → "
    "Виртуальная память, размер по выбору системы)."
)


def friendly(exc: BaseException) -> str:
    """One readable line for the Blender panel."""
    if isinstance(exc, UserError):
        return str(exc)
    if is_cuda_oom(exc):
        return "Не хватило видеопамяти. " + VRAM_HINT
    if is_ram_oom(exc):
        return RAM_HINT
    name = type(exc).__name__
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"Внутренняя ошибка сервера ({name}: {text}). Подробности — в логе сервера."
