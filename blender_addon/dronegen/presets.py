"""Prompt presets for typical drone show figures (English prompts work best with the models).

The style words (flat / 3D render / turnaround sheet) are added by the server (styles.py).
"""

DEFAULT_NEGATIVE = "photo, realistic, background scenery, shadow, text, watermark, frame, multiple objects, cropped"

PRESETS = [  # (key, menu label, prompt)
    ("TULPAR", "Тулпар", "Tulpar, the winged horse of Kazakh legend, galloping with wings spread wide, side view"),
    ("BERKUT", "Беркут", "golden eagle (berkut) with wings fully spread, front view, symmetrical"),
    (
        "SHANYRAK",
        "Шанырак",
        "Kazakh shanyrak, the round crown of a yurt: a circle with crossed arches inside, top view, symmetrical ornament",
    ),
    (
        "GOLDEN_MAN",
        "Золотой человек",
        "the Golden Man, ancient Saka warrior in golden armor with a tall pointed headdress, full body, front view",
    ),
    ("IRBIS", "Снежный барс", "snow leopard (irbis) leaping, side view"),
    (
        "DOMBRA",
        "Домбра",
        "Kazakh dombra, a long-necked lute with exactly two strings and two tuning pegs: a pear-shaped wooden body "
        "with a flat soundboard and a rounded back, a long slender neck with frets",
    ),
]

PROMPTS = {key: prompt for key, _label, prompt in PRESETS}
