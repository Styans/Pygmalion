"""Templates: exact models of known objects, built from named parameters.

A template is a function ``build(values) -> trimesh.Trimesh`` (vertex colours per part) plus a
table of parameters with ranges, so a request like «гриф длиннее» becomes a number the chat
assistant can set. Add a template: write the build function, describe the parameters,
``register`` it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..errors import UserError


@dataclass(frozen=True)
class Param:
    default: float
    lo: float
    hi: float
    title: str  # Russian, for the chat
    unit: str = ""
    integer: bool = False


@dataclass(frozen=True)
class Template:
    key: str
    title: str  # Russian name of the object
    params: dict[str, Param]
    build: Callable[[dict], object]
    about: str = ""  # one Russian sentence: what the drawing models
    names: tuple[str, ...] = field(default_factory=tuple)


TEMPLATES: dict[str, Template] = {}


def register(t: Template) -> Template:
    TEMPLATES[t.key] = t
    return t


def get(key: str) -> Template:
    try:
        return TEMPLATES[key]
    except KeyError:
        raise UserError(f"Нет чертежа «{key}». Есть: {', '.join(TEMPLATES)}") from None


def resolve(key: str, changes: dict[str, float] | None = None) -> dict[str, float]:
    """All parameter values of a template: defaults, then ``changes`` checked against the ranges."""
    t = get(key)
    values = {name: p.default for name, p in t.params.items()}
    for name, value in (changes or {}).items():
        if name not in t.params:
            raise UserError(f"У чертежа «{key}» нет параметра «{name}». Есть: {', '.join(t.params)}")
        p = t.params[name]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise UserError(f"Параметр «{name}» должен быть числом") from None
        if not p.lo <= value <= p.hi:
            raise UserError(f"{p.title} («{name}»): от {p.lo:g} до {p.hi:g}{(' ' + p.unit) if p.unit else ''}, а не {value:g}")
        values[name] = round(value) if p.integer else value
    return values


def build(key: str, changes: dict[str, float] | None = None):
    """(mesh, values) of a template."""
    values = resolve(key, changes)
    return get(key).build(values), values


def describe(key: str) -> str:
    """The parameter table as text for the chat assistant."""
    t = get(key)
    rows = [f"- {name}: {p.title}, по умолчанию {p.default:g}, от {p.lo:g} до {p.hi:g}{(' ' + p.unit) if p.unit else ''}"
            for name, p in t.params.items()]
    return f"Чертёж «{t.title}» ({key}). {t.about}\nПараметры:\n" + "\n".join(rows)
