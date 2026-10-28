"""Motions for skeletons with roles: a catalog and the keys of each motion for a given skeleton (``bake``).

A motion speaks of parts and directions of the body, never of bone names or rolls: «raise the wings»,
«swing the legs forward», «turn the head left». Every bone turn is an axis in the body's frame (forward,
left, up) and an angle; ``bake`` turns it into the bone's own (local) quaternion with the bone's rest matrix,
so a motion fits any skeleton from ``rigging.fit``, a neural rig or a skeleton edited by hand.

Turns are joint angles: a child bone's turn adds to its parent's (the tip of a wing flaps more than its
root when each segment turns a little). Three kinds:

* ``cycle`` repeats every ``period`` seconds (a flap, a gallop, breathing): ``t`` runs 0..1, and the last
  key equals the first;
* ``pose`` is one held key (wings up, rearing): Blender fades into it and out of it, which makes transitions;
* ``once`` plays once over ``period`` (a sword swing, a jump).

Directions: «up» lifts the tip of a part, «forward» swings a hanging limb forward, positive yaw turns
towards the figure's left. Drone shows are watched from far away and drones are slow, so the default
periods are slow too (a wing beat of 4 s); the safety check in Blender tells when a motion is too fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import UserError

TAU = 2.0 * np.pi
SAMPLES = 24  # keys per cycle: smooth enough with Blender's curves, few enough to edit by hand


# -- the skeleton ----------------------------------------------------------------------------
@dataclass
class Bone:
    name: str
    role: str
    side: str
    chain: str
    segment: int
    head: np.ndarray
    tail: np.ndarray
    rest: np.ndarray  # 3×3: the bone's axes in the armature's space (columns X, Y along the bone, Z)
    parent: str | None = None

    @property
    def direction(self) -> np.ndarray:
        d = self.tail - self.head
        return d / max(np.linalg.norm(d), 1e-12)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.tail - self.head))


@dataclass
class Skeleton:
    bones: list[Bone]
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray
    size: float
    by_name: dict = field(default_factory=dict)

    def __post_init__(self):
        self.by_name = {b.name: b for b in self.bones}

    def chains(self, *roles: str) -> list[list[Bone]]:
        """Bones of these roles grouped by chain, each from the body out."""
        out: dict[str, list[Bone]] = {}
        for b in self.bones:
            if b.role in roles or (b.role == "head" and "neck" in roles):
                out.setdefault(b.chain, []).append(b)
        return [sorted(c, key=lambda b: b.segment) for c in out.values()]

    def roles(self) -> set[str]:
        return {b.role for b in self.bones}

    def side(self, b: Bone) -> np.ndarray:
        """The figure's left for a left part, its right for a right one, nothing in the middle."""
        return self.left if b.side == "L" else -self.left if b.side == "R" else np.zeros(3)

    @property
    def pitch(self) -> np.ndarray:  # turning about it lifts the nose and swings a hanging foot forward
        return -self.left

    def raise_axis(self, b: Bone) -> np.ndarray:
        """Turning about it lifts the tip of a part that stands out (a wing, a raised arm) towards up."""
        a = np.cross(b.direction, self.up)
        if np.linalg.norm(a) < 0.2:  # pointing straight up or down: lift it forward
            return self.pitch
        return a / np.linalg.norm(a)

    def swing_axis(self, b: Bone) -> np.ndarray:
        """Turning about it swings the tip forward (an arm, a leg)."""
        a = np.cross(b.direction, self.forward)
        if np.linalg.norm(a) < 0.2:
            return self.up
        return a / np.linalg.norm(a)

    @property
    def root(self) -> Bone | None:
        return next((b for b in self.bones if b.role == "root"), None)


def skeleton(data: dict) -> Skeleton:
    """From the JSON the Blender add-on sends (or ``rigging.fit`` plus ``rest_matrices``)."""
    try:
        frame = data["frame"]
        bones = [Bone(b["name"], b["role"], b.get("side", ""), b.get("chain", b["name"]), int(b.get("segment", 1)),
                      np.asarray(b["head"], np.float64), np.asarray(b["tail"], np.float64),
                      np.asarray(b["matrix"], np.float64).reshape(3, 3), b.get("parent"))
                 for b in data["bones"]]
        sk = Skeleton(bones, _unit(frame["forward"]), _unit(frame["left"]), _unit(frame.get("up", (0, 0, 1))),
                      float(data.get("size") or 1.0))
    except (KeyError, TypeError, ValueError) as e:
        raise UserError(f"Скелет: неполные данные ({e})") from None
    if not bones:
        raise UserError("Скелет пустой")
    return sk


def rest_matrices(bones: list[dict], frame: dict) -> list[dict]:
    """Rest matrices as the Blender add-on makes them (the bone's Z faces up, forward for upright bones), for
    skeletons that never went through Blender (tests, a preview in the Studio)."""
    up, forward = _unit(frame.get("up", (0, 0, 1))), _unit(frame["forward"])
    out = []
    for b in bones:
        y = _unit(np.asarray(b["tail"], np.float64) - np.asarray(b["head"], np.float64))
        hint = forward if abs(float(y @ up)) > 0.7 else up
        z = _unit(hint - (hint @ y) * y)
        x = np.cross(y, z)
        out.append({**b, "matrix": np.stack([x, y, z], 1).tolist()})
    return out


# -- the catalog -----------------------------------------------------------------------------
@dataclass
class Param:
    title: str
    default: float
    lo: float
    hi: float
    unit: str = ""


@dataclass
class Motion:
    key: str
    title: str
    kind: str  # cycle | pose | once
    roles: tuple  # the motion needs at least one bone of these roles
    period: float  # seconds: one cycle, or the whole of a one-off; nothing for a pose
    about: str
    params: dict
    fn: object
    group: str = ""

    def resolve(self, values: dict | None) -> dict:
        out = {}
        values = dict(values or {})
        for name, p in self.params.items():
            v = values.pop(name, p.default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise UserError(f"«{self.title}»: параметр «{p.title}» должен быть числом") from None
            if not p.lo <= v <= p.hi:
                raise UserError(f"«{self.title}»: «{p.title}» от {p.lo:g} до {p.hi:g}{(' ' + p.unit) if p.unit else ''}")
            out[name] = v
        if values:
            raise UserError(f"«{self.title}»: нет параметра «{next(iter(values))}». Есть: {', '.join(self.params) or 'нет'}")
        return out

    def describe(self) -> dict:
        return {"title": self.title, "kind": self.kind, "roles": list(self.roles), "period": self.period,
                "about": self.about, "group": self.group,
                "params": {n: {"title": p.title, "default": p.default, "min": p.lo, "max": p.hi, "unit": p.unit}
                           for n, p in self.params.items()}}


MOTIONS: dict[str, Motion] = {}
ROLE_WORDS = {"wing": "крыльев", "leg": "ног", "leg_front": "передних ног", "leg_hind": "задних ног", "arm": "рук",
              "tail": "хвоста", "neck": "шеи", "head": "головы", "spine": "тела", "root": "корневой кости",
              "extra": "дополнительных частей", "prop": "предмета в руке"}


def motion(key, title, kind, roles, period=0.0, about="", group="", **params):
    def deco(fn):
        MOTIONS[key] = Motion(key, title, kind, tuple(roles), period, about, params, fn, group)
        return fn

    return deco


def catalog() -> dict:
    return {k: m.describe() for k, m in MOTIONS.items()}


def fits(m: Motion, roles: set[str]) -> bool:
    return any(r in roles for r in m.roles)


# -- turns -----------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot(axis, degrees: float) -> np.ndarray:
    """A quaternion (w, x, y, z) turning ``degrees`` about ``axis`` (right-handed)."""
    half = np.radians(degrees) / 2.0
    return np.concatenate([[np.cos(half)], np.sin(half) * _unit(axis)])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_q(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


class Pose:
    """Turns (armature space) of bones and a shift of the whole figure, for one moment."""

    def __init__(self):
        self.turns: dict[str, np.ndarray] = {}
        self.shift = np.zeros(3)

    def turn(self, bone: Bone | None, axis, degrees: float) -> None:
        if bone is None or abs(degrees) < 1e-9:
            return
        q = rot(axis, degrees)
        self.turns[bone.name] = qmul(q, self.turns[bone.name]) if bone.name in self.turns else q


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# -- cycles ----------------------------------------------------------------------------------
@motion("wing_flap", "Взмах крыльями", "cycle", ("wing",), 4.0, "Крылья поднимаются и опускаются, кончики чуть "
        "запаздывают; тело немного поднимается на взмахе вниз", "Крылья",
        amplitude=Param("Размах", 35, 2, 80, "°"), lag=Param("Запаздывание кончиков", 0.08, 0, 0.3, "цикла"),
        bob=Param("Подъём тела", 0.02, 0, 0.1, "размера"))
def _wing_flap(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for chain in sk.chains("wing"):
        axis = np.cross(sk.side(chain[0]), sk.up)
        if np.linalg.norm(axis) < 1e-6:  # a wing in the middle (a single fin): flap it up and down
            axis = sk.pitch
        for k, b in enumerate(chain):
            share = (1.0, 0.45, 0.3)[min(k, 2)]
            pose.turn(b, axis, p["amplitude"] * share * np.sin(TAU * (t - k * p["lag"])))
    pose.shift = sk.up * p["bob"] * sk.size * -np.sin(TAU * t)
    return pose


@motion("wing_glide", "Парение", "cycle", ("wing",), 6.0, "Крылья раскрыты и едва покачиваются, как у парящей птицы",
        "Крылья", amplitude=Param("Размах", 6, 1, 30, "°"))
def _wing_glide(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for chain in sk.chains("wing"):
        axis = np.cross(sk.side(chain[0]), sk.up)
        if np.linalg.norm(axis) < 1e-6:
            axis = sk.pitch
        for k, b in enumerate(chain):
            pose.turn(b, axis, p["amplitude"] * (1.0 if k == 0 else 0.4) * np.sin(TAU * (t - 0.15 * k)))
    pose.shift = sk.up * 0.01 * sk.size * np.sin(TAU * t)
    return pose


GAITS = {  # where in the cycle each leg is (share of the cycle): (front or hind or leg, side)
    "walk": {("hind", "L"): 0.0, ("front", "L"): 0.25, ("hind", "R"): 0.5, ("front", "R"): 0.75},
    "trot": {("hind", "L"): 0.0, ("front", "R"): 0.0, ("hind", "R"): 0.5, ("front", "L"): 0.5},
    "gallop": {("hind", "L"): 0.0, ("hind", "R"): 0.1, ("front", "L"): 0.45, ("front", "R"): 0.55},
}


def _legs(sk: Skeleton, p: dict, t: float, phases: dict, pose: Pose) -> None:
    for chain in sk.chains("leg_front", "leg_hind", "leg"):
        b0 = chain[0]
        end = "front" if b0.role == "leg_front" else "hind" if b0.role == "leg_hind" else "leg"
        phase = phases.get((end, b0.side), phases.get(("leg", b0.side), 0.0))
        a = TAU * (t - phase)
        lift = max(0.0, np.cos(a))  # the foot is up while the leg swings forward
        pose.turn(b0, sk.pitch, p["stride"] * np.sin(a))
        if len(chain) > 1:
            pose.turn(chain[1], sk.pitch, -p["knee"] * lift)
        if len(chain) > 2:
            pose.turn(chain[2], sk.pitch, 0.5 * p["knee"] * lift)


def _gait(name: str, rock: float, bumps: int):
    def fn(sk: Skeleton, p: dict, t: float) -> Pose:
        pose = Pose()
        _legs(sk, p, t, GAITS[name], pose)
        spine = [b for c in sk.chains("spine") for b in c]
        for b in spine:  # the body rocks, the neck and the tail answer it
            pose.turn(b, sk.pitch, rock * np.sin(TAU * t) / max(len(spine), 1))
        for c in sk.chains("neck"):
            pose.turn(c[0], sk.pitch, -0.7 * rock * np.sin(TAU * t))
        for c in sk.chains("tail"):
            for k, b in enumerate(c):
                pose.turn(b, sk.pitch, 0.5 * rock * np.sin(TAU * (t - 0.12 * (k + 1))))
        wave = np.abs(np.sin(TAU * t)) if bumps == 2 else 0.5 + 0.5 * np.sin(TAU * t)
        pose.shift = sk.up * p["bob"] * sk.size * wave
        return pose

    return fn


_LEG_PARAMS = dict(stride=Param("Шаг ног", 25, 5, 60, "°"), knee=Param("Сгиб колен", 40, 0, 90, "°"),
                   bob=Param("Подскок тела", 0.015, 0, 0.1, "размера"))
motion("walk", "Шаг", "cycle", ("leg_front", "leg_hind"), 3.0, "Четыре ноги по очереди: задняя левая, передняя левая, "
       "задняя правая, передняя правая", "Ноги", **_LEG_PARAMS)(_gait("walk", 3, 2))
motion("trot", "Рысь", "cycle", ("leg_front", "leg_hind"), 2.4, "Ноги парами по диагонали", "Ноги",
       **{**_LEG_PARAMS, "stride": Param("Шаг ног", 30, 5, 60, "°")})(_gait("trot", 4, 2))
motion("gallop", "Галоп", "cycle", ("leg_front", "leg_hind"), 2.0, "Задние ноги толкают, передние тянут; тело "
       "раскачивается, голова и хвост отвечают", "Ноги",
       **{**_LEG_PARAMS, "stride": Param("Шаг ног", 40, 5, 70, "°"), "bob": Param("Подскок тела", 0.03, 0, 0.1, "размера")}
       )(_gait("gallop", 8, 1))


@motion("walk_biped", "Шаг (двуногий)", "cycle", ("leg",), 2.4, "Ноги по очереди, руки качаются навстречу", "Ноги",
        stride=Param("Шаг ног", 25, 5, 60, "°"), knee=Param("Сгиб колен", 45, 0, 90, "°"),
        arms=Param("Мах руками", 20, 0, 60, "°"), bob=Param("Подскок тела", 0.01, 0, 0.1, "размера"))
def _walk_biped(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    _legs(sk, p, t, {("leg", "L"): 0.0, ("leg", "R"): 0.5}, pose)
    for c in sk.chains("arm"):
        phase = 0.5 if c[0].side == "L" else 0.0  # against the leg of its side
        pose.turn(c[0], sk.swing_axis(c[0]), p["arms"] * np.sin(TAU * (t - phase)))
    pose.shift = sk.up * p["bob"] * sk.size * np.abs(np.sin(TAU * t))
    return pose


@motion("tail_wave", "Волна хвостом", "cycle", ("tail",), 4.0, "Хвост волной из стороны в сторону, кончик запаздывает",
        "Хвост", amplitude=Param("Размах", 20, 2, 60, "°"), lag=Param("Запаздывание кончика", 0.12, 0, 0.3, "цикла"),
        vertical=Param("Вверх-вниз вместо вбок (0 или 1)", 0, 0, 1))
def _tail_wave(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    axis = sk.pitch if p["vertical"] >= 0.5 else sk.up
    for c in sk.chains("tail"):
        for k, b in enumerate(c):
            share = 0.5 + 0.5 * k / max(len(c) - 1, 1)
            pose.turn(b, axis, p["amplitude"] * share * np.sin(TAU * (t - k * p["lag"])) / max(len(c) ** 0.5, 1))
    return pose


@motion("breathe", "Дыхание", "cycle", ("spine",), 5.0, "Грудь чуть поднимается и опускается: фигура живая, даже "
        "когда стоит", "Тело", amplitude=Param("Размах", 3, 0.5, 15, "°"))
def _breathe(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("spine"):
        for b in c:
            pose.turn(b, sk.pitch, p["amplitude"] * np.sin(TAU * t) / len(c))
    for c in sk.chains("neck"):
        pose.turn(c[0], sk.pitch, -0.5 * p["amplitude"] * np.sin(TAU * t))
    return pose


@motion("look_around", "Взгляд по сторонам", "cycle", ("neck", "head"), 8.0, "Голова медленно поворачивается влево и "
        "вправо", "Голова", amplitude=Param("Поворот", 30, 5, 80, "°"))
def _look_around(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("neck"):
        for b in c:
            pose.turn(b, sk.up, p["amplitude"] * (0.6 if b.role == "head" else 0.4 / max(len(c) - 1, 1)) * np.sin(TAU * t))
    return pose


@motion("head_nod", "Кивок", "cycle", ("neck", "head"), 3.0, "Голова кивает вверх-вниз", "Голова",
        amplitude=Param("Размах", 15, 2, 45, "°"))
def _head_nod(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("neck"):
        for b in c:
            pose.turn(b, sk.pitch, p["amplitude"] * (0.6 if b.role == "head" else 0.4 / max(len(c) - 1, 1)) * np.sin(TAU * t))
    return pose


@motion("sway", "Покачивание", "cycle", ("spine",), 6.0, "Вся фигура плавно покачивается с боку на бок (подходит и "
        "для предметов)", "Тело", amplitude=Param("Размах", 6, 1, 30, "°"))
def _sway(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("spine"):
        for b in c:
            pose.turn(b, sk.forward, p["amplitude"] * np.sin(TAU * t) / len(c))
    return pose


@motion("wiggle", "Шевеление частей", "cycle", ("extra",), 3.0, "Уши, рога, гребни и прочие части слегка шевелятся",
        "Тело", amplitude=Param("Размах", 10, 1, 40, "°"))
def _wiggle(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for i, c in enumerate(sk.chains("extra")):
        for k, b in enumerate(c):
            pose.turn(b, sk.raise_axis(b), p["amplitude"] * np.sin(TAU * (t - 0.17 * i - 0.1 * k)))
    return pose


@motion("spin", "Вращение", "cycle", ("root",), 12.0, "Фигура поворачивается вокруг себя (вертикальной оси)",
        "Вся фигура", turns=Param("Оборотов за цикл", 1, -3, 3))
def _spin(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    pose.turn(sk.root, sk.up, 360.0 * p["turns"] * t)
    return pose


@motion("bob", "Подъём и опускание", "cycle", ("root",), 4.0, "Фигура плавно поднимается и опускается целиком",
        "Вся фигура", amplitude=Param("Высота", 0.05, 0.005, 0.5, "размера"))
def _bob(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    pose.shift = sk.up * p["amplitude"] * sk.size * np.sin(TAU * t)
    return pose


# -- poses -----------------------------------------------------------------------------------
@motion("wings_up", "Крылья вверх", "pose", ("wing",), about="Крылья подняты", group="Крылья",
        angle=Param("Угол", 50, 5, 90, "°"))
def _wings_up(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("wing"):
        axis = np.cross(sk.side(c[0]), sk.up)
        axis = axis if np.linalg.norm(axis) > 1e-6 else sk.pitch
        pose.turn(c[0], axis, p["angle"])
        if len(c) > 1:
            pose.turn(c[1], axis, 0.3 * p["angle"])
    return pose


@motion("wings_down", "Крылья вниз", "pose", ("wing",), about="Крылья опущены", group="Крылья",
        angle=Param("Угол", 30, 5, 80, "°"))
def _wings_down(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("wing"):
        axis = np.cross(sk.side(c[0]), sk.up)
        axis = axis if np.linalg.norm(axis) > 1e-6 else sk.pitch
        pose.turn(c[0], axis, -p["angle"])
        if len(c) > 1:
            pose.turn(c[1], axis, -0.2 * p["angle"])
    return pose


@motion("wings_folded", "Крылья сложены", "pose", ("wing",), about="Крылья отведены назад и сложены вдоль тела",
        group="Крылья", angle=Param("Угол", 70, 10, 110, "°"))
def _wings_folded(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("wing"):
        back = np.cross(sk.side(c[0]), -sk.forward)  # swings the tip backwards
        if np.linalg.norm(back) < 1e-6:
            continue
        lift = np.cross(sk.side(c[0]), sk.up)
        pose.turn(c[0], back, p["angle"])
        pose.turn(c[0], lift, -15)
        if len(c) > 1:
            pose.turn(c[1], back, -1.2 * p["angle"])
        if len(c) > 2:
            pose.turn(c[2], back, 0.8 * p["angle"])
    return pose


@motion("rear_up", "На дыбы", "pose", ("leg_front", "leg_hind"), about="Передняя часть поднимается, передние ноги "
        "поджаты, задние стоят", group="Ноги", angle=Param("Подъём тела", 35, 10, 60, "°"))
def _rear_up(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    spine = [b for c in sk.chains("spine") for b in c]
    if spine:
        pose.turn(spine[0], sk.pitch, p["angle"])
    for c in sk.chains("leg_hind"):  # they hang on the hips, which turned: back to the ground
        pose.turn(c[0], sk.pitch, -p["angle"])
    for c in sk.chains("leg_front"):
        pose.turn(c[0], sk.pitch, 60 - p["angle"])
        if len(c) > 1:
            pose.turn(c[1], sk.pitch, -90)
    for c in sk.chains("neck"):
        pose.turn(c[0], sk.pitch, 10)
    for c in sk.chains("tail"):
        pose.turn(c[0], sk.pitch, -0.5 * p["angle"])
    return pose


def _head_turn(yaw: float, pitch: float):
    def fn(sk: Skeleton, p: dict, t: float) -> Pose:
        pose = Pose()
        for c in sk.chains("neck"):
            for b in c:
                share = 0.6 if b.role == "head" else 0.4 / max(len(c) - 1, 1)
                pose.turn(b, sk.up, yaw * p["angle"] * share)
                pose.turn(b, sk.pitch, pitch * p["angle"] * share)
        return pose

    return fn


for _key, _title, _yaw, _pitch in (("head_left", "Голова влево", 1, 0), ("head_right", "Голова вправо", -1, 0),
                                   ("head_up", "Голова вверх", 0, 1), ("head_down", "Голова вниз", 0, -1)):
    motion(_key, _title, "pose", ("neck", "head"), about=_title, group="Голова",
           angle=Param("Угол", 35, 5, 80, "°"))(_head_turn(_yaw, _pitch))


@motion("arm_raise", "Рука вверх", "pose", ("arm",), about="Рука поднята (side: 1 — левая, -1 — правая, 0 — обе)",
        group="Руки", angle=Param("Угол", 70, 10, 170, "°"), side=Param("Какая рука", -1, -1, 1))
def _arm_raise(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in _arms(sk, p["side"]):
        pose.turn(c[0], sk.raise_axis(c[0]), p["angle"])
    return pose


@motion("tail_up", "Хвост вверх", "pose", ("tail",), about="Хвост поднят дугой", group="Хвост",
        angle=Param("Угол", 40, 5, 90, "°"))
def _tail_up(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    for c in sk.chains("tail"):
        for b in c:
            pose.turn(b, sk.pitch, p["angle"] / len(c) * 1.5)
    return pose


# -- once ------------------------------------------------------------------------------------
def _arms(sk: Skeleton, side: float) -> list[list[Bone]]:
    chains = sk.chains("arm")
    if side >= 0.5:
        return [c for c in chains if c[0].side == "L"] or chains[:1]
    if side <= -0.5:
        return [c for c in chains if c[0].side == "R"] or chains[:1]
    return chains


@motion("sword_swing", "Взмах мечом", "once", ("arm",), 2.0, "Замах вверх-назад и удар вперёд-вниз той рукой, что "
        "держит предмет (side: 1 — левая, -1 — правая)", "Руки",
        angle=Param("Размах", 110, 30, 170, "°"), side=Param("Какая рука", -1, -1, 1))
def _sword_swing(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    if t < 0.35:  # the wind-up
        a = p["angle"] * _smooth(t / 0.35)
    elif t < 0.55:  # the strike: fast
        a = p["angle"] - 1.4 * p["angle"] * _smooth((t - 0.35) / 0.2)
    else:  # back to rest
        a = -0.4 * p["angle"] * (1 - _smooth((t - 0.55) / 0.45))
    for c in _arms(sk, p["side"]):
        pose.turn(c[0], sk.raise_axis(c[0]), a)
        if len(c) > 1:
            pose.turn(c[1], sk.swing_axis(c[1]), 0.25 * a)
    return pose


@motion("jump", "Прыжок", "once", ("root",), 2.0, "Фигура подпрыгивает, поджимая ноги", "Вся фигура",
        height=Param("Высота", 0.3, 0.05, 1.5, "размера"), tuck=Param("Поджать ноги", 40, 0, 90, "°"))
def _jump(sk: Skeleton, p: dict, t: float) -> Pose:
    pose = Pose()
    arc = 4 * t * (1 - t)
    pose.shift = sk.up * p["height"] * sk.size * arc
    for c in sk.chains("leg_front", "leg_hind", "leg"):
        front = c[0].role != "leg_hind"
        pose.turn(c[0], sk.pitch, (1 if front else -1) * p["tuck"] * arc)
        if len(c) > 1:
            pose.turn(c[1], sk.pitch, -(1 if front else -1) * 2 * p["tuck"] * arc)
    return pose


# -- baking ----------------------------------------------------------------------------------
def bake(sk: Skeleton, key: str, values: dict | None = None, samples: int = SAMPLES) -> dict:
    """Keys of a motion for a skeleton: times (0..1 of the period; 0..2 when ``span`` is 2, see below), each
    bone's local quaternions, the root's local shift. Only bones the motion turns are listed."""
    m = MOTIONS.get(key)
    if m is None:
        raise UserError(f"Нет движения «{key}». Есть: {', '.join(MOTIONS)}")
    if not fits(m, sk.roles()):
        need = " или ".join(ROLE_WORDS.get(r, r) for r in m.roles)
        raise UserError(f"«{m.title}»: у скелета нет {need}")
    p = m.resolve(values)
    samples = int(np.clip(samples, 4, 240))
    times = [0.0] if m.kind == "pose" else list(np.linspace(0.0, 1.0, samples + 1))
    poses = [m.fn(sk, p, float(t)) for t in times]
    names = sorted({n for pose in poses for n in pose.turns})
    turns = {}
    for name in names:
        b = sk.by_name[name]
        rest = _mat_q(b.rest)
        rest_inv = rest * np.array([1.0, -1.0, -1.0, -1.0])  # the conjugate: no turn stays exactly (1, 0, 0, 0)
        keys, prev = [], None
        for pose in poses:
            q = pose.turns.get(name, np.array([1.0, 0, 0, 0]))
            local = qmul(qmul(rest_inv, q), rest)
            if prev is None and local[0] < 0 or prev is not None and float(local @ prev) < 0:
                # q and -q are one turn, but Blender mixes layers with a part of a turn (fades, strength): -q would
                # go the long way round, almost 360°. Start on the near side (w ≥ 0), then no flips between keys
                local = -local
            keys.append(local)
            prev = local
        turns[name] = np.array(keys)
    shifts = np.array([pose.shift for pose in poses])
    span = 1
    if m.kind == "cycle":
        # a whole turn ends on the other side of the quaternion sphere (q and -q are one turn): such a loop has a
        # seam, so its keys go round twice and end where they began
        if any(float(k[-1] @ k[0]) < -0.999 for k in turns.values()):
            span = 2
            for name, k in turns.items():
                sign = -1.0 if float(k[-1] @ k[0]) < 0 else 1.0
                turns[name] = np.concatenate([k, sign * k[1:]])
            shifts = np.concatenate([shifts, shifts[1:]])
            times = times + [1.0 + t for t in times[1:]]
        for name, k in turns.items():
            if float(k[-1] @ k[0]) > 0.999:
                k[-1] = k[0]  # closed exactly: the loop has no seam
        shifts[-1] = shifts[0]
    bones = {name: [[round(float(x), 6) for x in q] for q in k] for name, k in turns.items()}
    root = None
    if sk.root is not None and np.abs(shifts).max() > 1e-9:
        local = shifts @ sk.root.rest  # armature -> the root bone's axes (rest is orthonormal)
        root = {"bone": sk.root.name, "location": [[round(float(x), 6) for x in v] for v in local]}
    return {"motion": key, "kind": m.kind, "period": m.period, "span": span, "params": p,
            "times": [round(float(t), 6) for t in times], "bones": bones, "root": root}


def posed_tails(sk: Skeleton, baked: dict, index: int) -> dict[str, np.ndarray]:
    """Where every bone's tail is at key ``index`` (armature space): forward kinematics, for tests and previews."""
    world: dict[str, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}

    def rest4(b: Bone) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = b.rest, b.head
        return m

    pending = list(sk.bones)
    while pending:
        rest_pending = []
        for b in pending:
            if b.parent and b.parent not in world:
                rest_pending.append(b)
                continue
            local = np.eye(4)
            if b.name in baked["bones"]:
                local[:3, :3] = _qmat(np.asarray(baked["bones"][b.name][index]))
            if baked.get("root") and baked["root"]["bone"] == b.name:
                local[:3, 3] = baked["root"]["location"][index]
            if b.parent:
                p = sk.by_name[b.parent]
                m = world[b.parent] @ np.linalg.inv(rest4(p)) @ rest4(b) @ local
            else:
                m = rest4(b) @ local
            world[b.name] = m
            out[b.name] = (m @ np.array([0.0, b.length, 0.0, 1.0]))[:3]
        if len(rest_pending) == len(pending):
            raise UserError("Скелет: кости ссылаются на несуществующих родителей")
        pending = rest_pending
    return out
