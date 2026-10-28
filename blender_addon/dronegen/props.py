"""Scene settings of the DroneGen panel (saved with the .blend) and the panel's own UI state."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
                       StringProperty)

from . import presets

IMAGE_BACKENDS = [
    ("zimage_turbo", "Z-Image-Turbo", "Быстрая модель 6B (GGUF), лучше всего понимает английский"),
    ("sdxl", "SDXL", "Запасная модель Stable Diffusion XL: медленнее, понимает негативный промпт"),
    ("mock", "Тест (без нейросети)", "Простая фигура по seed — проверить связку без видеокарты"),
]
MESH_BACKENDS = [
    ("hunyuan", "Hunyuan3D-mini", "Точная форма по одной объёмной картинке (нужна видеокарта)"),
    ("hunyuan_mv", "Hunyuan3D-mv", "Самая точная форма: картинка — лист из 4 видов (спереди | слева | сзади | справа)"),
    ("triposr", "TripoSR", "Быстрая, но грубая нейросеть: объём по одной картинке (нужна видеокарта)"),
    ("silhouette", "Силуэт → объём", "Без нейросети и видеокарты: «надутый» силуэт, ровный и предсказуемый"),
    ("mock", "Тест (сфера)", "Проверить связку"),
]
SIZES = [
    ("512x512", "512 × 512", "Квадрат"),
    ("768x768", "768 × 768", "Квадрат"),
    ("1024x1024", "1024 × 1024", "Квадрат"),
    ("768x1024", "768 × 1024", "Вертикально"),
    ("1024x768", "1024 × 768", "Горизонтально"),
    ("1920x576", "1920 × 576", "Лист из 4 видов"),
]
IMAGE_STYLES = [
    ("render", "Объёмный (для 3D)", "Затенённый 3D-рендер: лучше всего для Hunyuan3D и TripoSR"),
    ("sheet", "Лист из 4 видов", "Спереди | слева | сзади | справа на одной картинке — для Hunyuan3D-mv, самая точная форма"),
    ("flat", "Плоский (для силуэта)", "Чёткий плоский силуэт — для «Силуэт → объём»"),
    ("none", "Без стиля", "Промпт как есть"),
]
STYLE_SIZES = {"render": "768x1024", "sheet": "1920x576", "flat": "768x768"}
STYLE_MESH = {"render": "hunyuan", "sheet": "hunyuan_mv", "flat": "silhouette"}
MESH_STYLES = [
    ("as_is", "Как есть", "Поверхность как её построила нейросеть"),
    ("smooth", "Гладко", "Пересобрать поверхность: гладкая и цельная (мелкие детали смягчаются)"),
    ("lowpoly", "Low-poly", "Мало плоских граней, как у сделанной вручную low-poly модели"),
]
BODY_PLANS = [  # the server's rigging.BODY_PLANS
    ("auto", "Авто", "По форме модели: ноги — что стоит на земле, крылья — плоские части по бокам"),
    ("quadruped", "Четвероногое", "Конь, барс, волк: 4 ноги, голова, хвост"),
    ("winged_quadruped", "Крылатое четвероногое", "Тулпар, пегас, грифон: 4 ноги и 2 крыла"),
    ("bird", "Птица", "Беркут, орёл: 2 лапы, 2 крыла, хвост"),
    ("biped", "Двуногое с руками", "Человек, персонаж: 2 ноги, 2 руки (меч в руке — отдельная кость)"),
    ("winged_biped", "Двуногое с крыльями", "Ангел, дракон на задних лапах: 2 ноги, 2 руки, 2 крыла"),
    ("snake", "Без конечностей", "Змея, рыба: одно длинное тело"),
    ("rigid", "Предмет", "Домбра, машина: одна цепочка костей, чтобы качать и поворачивать целиком"),
]
RIG_METHODS = [
    ("template", "По форме модели", "Скелет по форме меша: быстро, без нейросети и видеокарты"),
    ("unirig", "Нейросеть UniRig", "Нейросеть подбирает скелет для любой формы (нужна видеокарта от 8 ГБ)"),
]
DEFAULT_STEPS = {"zimage_turbo": 9, "sdxl": 25, "mock": 8}
DEFAULT_RES = {"hunyuan": "384", "hunyuan_mv": "384", "triposr": "256", "silhouette": "192", "mock": "128"}


def _image_backend_changed(self, context):
    self.steps = DEFAULT_STEPS.get(self.image_backend, self.steps)


def _image_style_changed(self, context):
    if self.image_style in STYLE_SIZES:
        self.size = STYLE_SIZES[self.image_style]
        target = STYLE_MESH[self.image_style]
        if self.mesh_backend != target:
            self.mesh_backend = target


def _mesh_backend_changed(self, context):
    self.mc_resolution = DEFAULT_RES.get(self.mesh_backend, self.mc_resolution)
    if self.mesh_backend == "hunyuan_mv" and self.image_style != "sheet":
        self.image_style = "sheet"


class DroneGenSettings(bpy.types.PropertyGroup):
    # picture
    prompt: StringProperty(name="Промпт", description="Что нарисовать (лучше по-английски)", default="")
    image_style: EnumProperty(
        name="Стиль",
        description="Как рисовать: под какой способ 3D готовится картинка (сервер дописывает промпт)",
        items=IMAGE_STYLES,
        default="render",
        update=_image_style_changed,
    )
    negative: StringProperty(name="Негативный", description="Чего не должно быть (только SDXL)", default=presets.DEFAULT_NEGATIVE)
    image_backend: EnumProperty(name="Модель", items=IMAGE_BACKENDS, default="zimage_turbo", update=_image_backend_changed)
    size: EnumProperty(name="Размер", items=SIZES, default="768x1024")
    steps: IntProperty(name="Шаги", default=9, min=1, max=100)
    seed: IntProperty(name="Seed", description="−1 — случайный", default=-1, min=-1)
    # 3D
    mesh_backend: EnumProperty(name="Способ", items=MESH_BACKENDS, default="hunyuan", update=_mesh_backend_changed)
    source: EnumProperty(
        name="Картинка",
        items=[("GALLERY", "Из галереи", "Выбранная в галерее картинка"), ("FILE", "Файл", "Своя картинка с диска")],
        default="GALLERY",
    )
    image_file: StringProperty(name="Файл", subtype="FILE_PATH", default="")
    remove_bg: BoolProperty(name="Убрать фон", default=True)
    mc_resolution: EnumProperty(
        name="Разрешение",
        description="Сетка marching cubes: выше — детальнее, но дольше и больше памяти",
        items=[("128", "128", ""), ("192", "192", ""), ("256", "256", ""), ("320", "320", ""), ("384", "384", "")],
        default="384",
    )
    symmetry: BoolProperty(
        name="Симметрия",
        description="Только Hunyuan3D: сделать левую и правую половины одинаковыми (для симметричных фигур)",
        default=False,
    )
    mesh_style: EnumProperty(name="Поверхность", description="Только Hunyuan3D: обработка сетки", items=MESH_STYLES, default="as_is")
    depth: FloatProperty(
        name="Толщина",
        description="Только для силуэта: наибольшая толщина относительно размера фигуры",
        default=0.3, min=0.02, max=1.0,
    )
    target_size: FloatProperty(name="Размер в сцене", description="Наибольшая сторона меша в сцене", default=2.0, min=0.01,
                               unit="LENGTH")
    target_faces: IntProperty(name="Упростить до", description="Треугольников (0 — не упрощать)", default=0, min=0, soft_max=500000)
    # skeleton
    rig_plan: EnumProperty(name="План тела", description="Какие части искать; «Авто» решает по форме",
                           items=BODY_PLANS, default="auto")
    rig_method: EnumProperty(name="Метод", description="Как строить скелет", items=RIG_METHODS, default="template")
    rig_bind: BoolProperty(name="Сразу привязать", description="Привязать меш и точки дронов к скелету, как только он "
                           "будет готов (иначе — кнопкой «Привязать» после правки скелета)", default=True)

    def image_size(self):
        w, h = self.size.split("x")
        return int(w), int(h)

    def style(self) -> str | None:
        """The picture style sent to the server (it adds the matching words to the prompt)."""
        return None if self.image_style == "none" else self.image_style


PATH_SHAPES = [
    ("CIRCLE", "Круг", "Фигура летит по кругу, начиная со своего места и поворачивая налево"),
    ("LINE", "Прямо", "Фигура летит вперёд по прямой"),
    ("CURVE", "Своя кривая", "Путь — выбранная кривая (сдвинутая к исходному месту фигуры)"),
]


def _is_curve(self, obj):
    return obj.type == "CURVE"


def _effect_changed(self, context):
    """Another kind of effect: big enough for it, and named after it unless the user named the layer."""
    from . import fx

    titles = {item.name for item in self.bl_rna.properties["effect"].enum_items}
    if self.name in titles:
        self.name = self.bl_rna.properties["effect"].enum_items[self.effect].name
    self.size = max(self.size, fx.min_size(self.effect, context.scene))


class DroneGenLayer(bpy.types.PropertyGroup):
    """One layer of a figure's animation: a motion or a flight over a span of frames, mixed with the others."""

    name: StringProperty(name="Название", default="Слой")
    enabled: BoolProperty(name="Включён", default=True)
    kind: EnumProperty(name="Тип", items=[
        ("MOTION", "Движение", "Движение частей тела из каталога сервера"),
        ("PATH", "Полёт", "Фигура летит по пути"),
        ("SHAPE", "Эффект формы", "Точки дронов фигуры: волна, пульс, дрожь, рассыпаться и собраться"),
        ("EFFECT", "Эффект с дронами", "Отдельные дроны: звезда, кольцо, искры, брызги, след движения… Ждут в резерве "
                                       "за фигурой, прилетают к началу и улетают после"),
    ], default="MOTION")
    shape_fx: EnumProperty(name="Эффект", items=[
        ("WAVE", "Волна по телу", "Волна бежит по телу фигуры, точки поднимаются и опускаются"),
        ("PULSE", "Пульс", "Фигура дышит: то чуть больше, то чуть меньше"),
        ("JITTER", "Дрожь", "Точки чуть дрожат, как пламя или вода"),
        ("SCATTER", "Рассыпаться и собраться", "Точки разлетаются в стороны и собираются обратно в фигуру"),
    ], default="WAVE")
    fx_size: FloatProperty(name="Сила", description="Волна, дрожь: на сколько сдвигаются точки; пульс: доля размера; "
                           "рассыпаться: на сколько разлетаются", default=1.0, min=0.0, soft_max=50.0)
    effect: EnumProperty(name="Эффект", items=[
        ("star", "Звезда", "Контур звезды"), ("ring", "Кольцо", "Контур круга"), ("heart", "Сердце", "Контур сердца"),
        ("ripple", "Расходящееся кольцо", "Кольцо расходится от точки, как волна от удара копыта"),
        ("sparkles", "Искры", "Дроны, рассыпанные в шаре"), ("splash", "Брызги", "Дроны вылетают вверх и в стороны и падают"),
        ("trail", "След движения", "Дроны встают вдоль пути кончика кости (взмах меча, крыла, копыта) за время слоя"),
        ("model", "Модель", "Дроны на поверхности выбранной модели (например, из студии)"),
    ], default="star", update=_effect_changed)
    size: FloatProperty(name="Размер", description="Размер эффекта поперёк", default=10.0, min=0.1, soft_max=200.0,
                        unit="LENGTH")
    count: IntProperty(name="Дронов", description="Сколько дронов в эффекте; 0 — сколько поместится на безопасном "
                       "расстоянии", default=0, min=0, max=2000)
    anchor: EnumProperty(name="Где", items=[
        ("ABOVE", "Над фигурой", "Над головой фигуры (летит вместе с ней)"),
        ("BONE", "У части тела", "Там, где кончик выбранной части в начале слоя (летит вместе с фигурой)"),
        ("WORLD", "На месте", "Неподвижно в небе: там, где фигура была в начале слоя"),
    ], default="ABOVE")
    bone: StringProperty(name="Часть", description="Цепочка костей: её кончик — место эффекта или путь следа "
                         "(например prop.R — меч, wing.L — левое крыло)", default="")
    offset: FloatProperty(name="Выше", description="Поднять эффект над местом привязки", default=0.0, unit="LENGTH")
    travel: FloatProperty(name="Перелёт", description="Секунд на перелёт дронов из резерва на места и обратно",
                          default=3.0, min=0.2, soft_max=20.0, unit="TIME_ABSOLUTE")
    model: PointerProperty(name="Модель", type=bpy.types.Object, poll=lambda self, obj: obj.type == "MESH")
    motion: StringProperty(name="Движение", description="Ключ движения в каталоге сервера", default="")
    start: IntProperty(name="Начало", description="Кадр, с которого слой начинается", default=1)
    end: IntProperty(name="Конец", description="Кадр, на котором слой заканчивается", default=241)
    period: FloatProperty(name="Период", description="Секунд на один цикл (или на всё движение)", default=4.0,
                          min=0.2, soft_max=60.0, unit="TIME_ABSOLUTE")
    amplitude: FloatProperty(name="Размах", description="Во сколько раз больше или меньше обычного размаха движения",
                             default=1.0, min=0.05, soft_max=3.0)
    strength: FloatProperty(name="Сила", description="Насколько слой перекрывает исходную позу", default=1.0, min=0.0,
                            max=1.0, subtype="FACTOR")
    fade_in: FloatProperty(name="Вход", description="Секунд на плавное начало", default=1.0, min=0.0, soft_max=10.0,
                           unit="TIME_ABSOLUTE")
    fade_out: FloatProperty(name="Выход", description="Секунд на плавное окончание", default=1.0, min=0.0, soft_max=10.0,
                            unit="TIME_ABSOLUTE")
    side: EnumProperty(name="Сторона", items=[("R", "Правая", ""), ("L", "Левая", ""), ("BOTH", "Обе", "")], default="R",
                       description="Для движений одной руки")
    path_shape: EnumProperty(name="Путь", items=PATH_SHAPES, default="CIRCLE")
    radius: FloatProperty(name="Радиус / длина", description="Радиус круга или длина прямого пути", default=30.0,
                          min=0.1, soft_max=500.0, unit="LENGTH")
    laps: FloatProperty(name="Кругов", description="Сколько раз пройти путь за время слоя", default=1.0, min=0.05,
                        soft_max=10.0)
    curve: PointerProperty(name="Кривая", type=bpy.types.Object, poll=_is_curve)


class DroneGenUI(bpy.types.PropertyGroup):
    """Which folds of the panel are open (per session, not saved with the file)."""

    image_options: BoolProperty(name="Параметры картинки", default=False)
    mesh_options: BoolProperty(name="Параметры 3D", default=False)


classes = (DroneGenSettings, DroneGenUI, DroneGenLayer)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dronegen = PointerProperty(type=DroneGenSettings)
    bpy.types.WindowManager.dronegen_ui = PointerProperty(type=DroneGenUI)
    bpy.types.Object.dronegen_layers = CollectionProperty(type=DroneGenLayer)  # on the skeleton (armature)
    bpy.types.Object.dronegen_layer = IntProperty(name="Слой", default=0)


def unregister():
    del bpy.types.Object.dronegen_layer
    del bpy.types.Object.dronegen_layers
    del bpy.types.WindowManager.dronegen_ui
    del bpy.types.Scene.dronegen
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
