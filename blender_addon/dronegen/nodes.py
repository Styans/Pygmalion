"""Geometry Nodes groups for drone points, built in code (no .blend to ship): effects of shape and effects with
drones of their own.

Effects of shape move the figure's drone points on top of the skeleton (they come after the Armature modifier):
a wave running along the body, a pulse, a tremble, scattering and gathering back. Each has a ``Factor`` input
that DroneGen keys for the layer's span and fades.

An effect with drones of its own (``EFFECT``) keeps every drone at its waiting place in the reserve, flies it to
its place ``Travel`` seconds before ``Start``, and back after ``End``; meanwhile a ripple spreads (``Expand``) and
a splash flies (``dg_velocity`` and ``Gravity``). Places are point attributes: ``dg_reserve``, ``dg_shape``.
Everything runs on the scene's time, so nothing is keyed per frame. Vertex groups pass through the nodes, so
Skybrush's markers (pins to vertex groups) follow the moved points.
"""

from __future__ import annotations

import bpy

VERSION = 2  # bump when a group changes: an older group of the same name in a .blend is rebuilt
PREFIX = "DG· "


def _sock(value):
    return value if hasattr(value, "links") or hasattr(value, "is_linked") else None


class _Builder:
    def __init__(self, ng):
        self.ng = ng
        self.nodes, self.links = ng.nodes, ng.links

    def node(self, kind: str, **settings):
        n = self.nodes.new(kind)
        for k, v in settings.items():
            setattr(n, k, v)
        return n

    def feed(self, socket, value) -> None:
        if hasattr(value, "node"):  # an output socket
            self.links.new(value, socket)
        else:
            socket.default_value = value

    def math(self, op: str, a, b=0.0, c=0.0):
        n = self.node("ShaderNodeMath", operation=op)
        for sock, v in zip(n.inputs, (a, b, c)):
            self.feed(sock, v)
        return n.outputs[0]

    def vmath(self, op: str, a, b=(0.0, 0.0, 0.0)):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(n.inputs[0], a)
        self.feed(n.inputs[1], b)
        return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]

    def scale(self, v, s):
        n = self.node("ShaderNodeVectorMath", operation="SCALE")
        self.feed(n.inputs[0], v)
        self.feed(n.inputs["Scale"], s)
        return n.outputs["Vector"]

    def ramp(self, value, lo, hi, out_lo=0.0, out_hi=1.0):
        """Smoothstep from ``lo`` to ``hi``, clamped."""
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        for name, v in (("Value", value), ("From Min", lo), ("From Max", hi), ("To Min", out_lo), ("To Max", out_hi)):
            self.feed(n.inputs[name], v)
        return n.outputs["Result"]

    def attribute(self, name: str, kind: str = "FLOAT_VECTOR"):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=kind)
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def seconds(self):
        return self.node("GeometryNodeInputSceneTime").outputs["Seconds"]

    def position(self):
        return self.node("GeometryNodeInputPosition").outputs["Position"]


def _group(name: str, inputs: list[tuple[str, str, object]], build) -> bpy.types.NodeTree:
    """A node group ``name`` with Geometry in and out and these inputs (name, socket type, default)."""
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.get("dg_version") == VERSION:
        return ng
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for sock_name, kind, default in inputs:
        s = ng.interface.new_socket(sock_name, in_out="INPUT", socket_type=kind)
        if default is not None:
            s.default_value = default
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    b = _Builder(ng)
    gi, go = b.node("NodeGroupInput"), b.node("NodeGroupOutput")
    offset_or_place = build(b, gi)
    sp = b.node("GeometryNodeSetPosition")
    b.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    kind, socket = offset_or_place
    b.links.new(socket, sp.inputs["Offset" if kind == "offset" else "Position"])
    b.links.new(sp.outputs["Geometry"], go.inputs["Geometry"])
    ng["dg_version"] = VERSION
    return ng


# -- effects of shape -------------------------------------------------------------------------
def _wave(b: _Builder, gi):
    along = b.math("DIVIDE", b.vmath("DOT_PRODUCT", b.position(), gi.outputs["Travel"]), gi.outputs["Wavelength"])
    cycle = b.math("DIVIDE", b.seconds(), gi.outputs["Period"])
    s = b.math("SINE", b.math("MULTIPLY", b.math("SUBTRACT", cycle, along), 6.283185307))
    k = b.math("MULTIPLY", b.math("MULTIPLY", s, gi.outputs["Amplitude"]), gi.outputs["Factor"])
    return "offset", b.scale(gi.outputs["Direction"], k)


def _pulse(b: _Builder, gi):
    s = b.math("SINE", b.math("MULTIPLY", b.math("DIVIDE", b.seconds(), gi.outputs["Period"]), 6.283185307))
    k = b.math("MULTIPLY", b.math("MULTIPLY", s, gi.outputs["Amplitude"]), gi.outputs["Factor"])
    return "offset", b.scale(b.vmath("SUBTRACT", b.position(), gi.outputs["Center"]), k)


def _jitter(b: _Builder, gi):
    noise = b.node("ShaderNodeTexNoise", noise_dimensions="4D")
    b.feed(noise.inputs["Vector"], b.scale(b.position(), gi.outputs["Scale"]))
    b.feed(noise.inputs["W"], b.math("MULTIPLY", b.seconds(), gi.outputs["Speed"]))
    centred = b.vmath("SUBTRACT", noise.outputs["Color"], (0.5, 0.5, 0.5))
    k = b.math("MULTIPLY", b.math("MULTIPLY", gi.outputs["Amplitude"], 2.0), gi.outputs["Factor"])
    return "offset", b.scale(centred, k)


def _scatter(b: _Builder, gi):
    rnd = b.node("FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
    rnd.inputs["Min"].default_value = (-1.0, -1.0, -1.0)
    rnd.inputs["Max"].default_value = (1.0, 1.0, 1.0)
    b.feed(rnd.inputs["Seed"], gi.outputs["Seed"])
    direction = b.vmath("NORMALIZE", rnd.outputs["Value"])
    return "offset", b.scale(direction, b.math("MULTIPLY", gi.outputs["Radius"], gi.outputs["Factor"]))


SHAPE_FX = {  # kind -> (title, about, inputs, builder)
    "WAVE": ("Волна по телу", "Волна бежит по телу фигуры, точки поднимаются и опускаются",
             [("Factor", "NodeSocketFloat", 0.0), ("Amplitude", "NodeSocketFloat", 0.5),
              ("Period", "NodeSocketFloat", 3.0), ("Wavelength", "NodeSocketFloat", 5.0),
              ("Direction", "NodeSocketVector", (0.0, 0.0, 1.0)), ("Travel", "NodeSocketVector", (0.0, -1.0, 0.0))],
             _wave),
    "PULSE": ("Пульс", "Фигура дышит: то чуть больше, то чуть меньше",
              [("Factor", "NodeSocketFloat", 0.0), ("Amplitude", "NodeSocketFloat", 0.1),
               ("Period", "NodeSocketFloat", 3.0), ("Center", "NodeSocketVector", (0.0, 0.0, 0.0))], _pulse),
    "JITTER": ("Дрожь", "Точки чуть дрожат, как пламя или вода",
               [("Factor", "NodeSocketFloat", 0.0), ("Amplitude", "NodeSocketFloat", 0.3),
                ("Speed", "NodeSocketFloat", 0.5), ("Scale", "NodeSocketFloat", 0.3)], _jitter),
    "SCATTER": ("Рассыпаться и собраться", "Точки разлетаются в стороны и собираются обратно в фигуру",
                [("Factor", "NodeSocketFloat", 0.0), ("Radius", "NodeSocketFloat", 5.0),
                 ("Seed", "NodeSocketInt", 0)], _scatter),
}


def shape_group(kind: str) -> bpy.types.NodeTree:
    title, _about, inputs, builder = SHAPE_FX[kind]
    return _group(f"{PREFIX}эффект: {title}", inputs, builder)


# -- effects with drones of their own ---------------------------------------------------------
def _effect(b: _Builder, gi):
    t = b.seconds()
    start, end, travel = gi.outputs["Start"], gi.outputs["End"], gi.outputs["Travel"]
    span = b.math("MAXIMUM", b.math("SUBTRACT", end, start), 1e-3)
    # time into the effect, eased: the drones gather speed after Start and stand still at End, before flying back
    tau = b.math("MULTIPLY", b.ramp(t, start, b.math("ADD", start, span)), span)
    grow = b.math("ADD", b.math("MULTIPLY", gi.outputs["Expand"], b.math("DIVIDE", tau, span)), 1.0)
    flight = b.vmath("ADD", b.scale(b.attribute("dg_velocity"), tau),
                     b.scale((0.0, 0.0, -0.5), b.math("MULTIPLY", gi.outputs["Gravity"], b.math("MULTIPLY", tau, tau))))
    place = b.vmath("ADD", b.scale(b.attribute("dg_shape"), grow), flight)
    arrive = b.ramp(t, b.math("SUBTRACT", start, travel), start)
    leave = b.ramp(t, end, b.math("ADD", end, travel), 1.0, 0.0)
    f = b.math("MINIMUM", arrive, leave)
    reserve = b.attribute("dg_reserve")
    return "place", b.vmath("ADD", reserve, b.scale(b.vmath("SUBTRACT", place, reserve), f))


EFFECT_INPUTS = [("Start", "NodeSocketFloat", 0.0), ("End", "NodeSocketFloat", 5.0), ("Travel", "NodeSocketFloat", 2.0),
                 ("Expand", "NodeSocketFloat", 0.0), ("Gravity", "NodeSocketFloat", 0.0)]


def effect_group() -> bpy.types.NodeTree:
    return _group(f"{PREFIX}эффект с дронами", EFFECT_INPUTS, _effect)


def set_input(mod, name: str, value) -> str:
    """Set a modifier's group input by its name; returns the data path of the value on the object (for keys).
    Blender 5 keeps the values in ``mod.properties.inputs.<id>.value``, Blender 4 in ``mod["<id>"]``."""
    ident = mod.node_group.interface.items_tree[name].identifier
    inputs = getattr(getattr(mod, "properties", None), "inputs", None)
    if inputs is not None and hasattr(inputs, ident):
        getattr(inputs, ident).value = value
        return f'modifiers["{mod.name}"].properties.inputs.{ident}.value'
    mod[ident] = value
    return f'modifiers["{mod.name}"]["{ident}"]'
