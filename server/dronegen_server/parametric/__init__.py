"""Exact models of known objects built from a drawing (parameters), not drawn by a neural network."""

from . import baiterek, dombra, shanyrak, yurt  # noqa: F401 - importing a module registers its template
from .registry import TEMPLATES, Param, Template, build, describe, get, resolve  # noqa: F401
