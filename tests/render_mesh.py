"""Render meshes the way the audience sees them: Front view, vertex colours (Workbench).

    blender -b --factory-startup --python tests/render_mesh.py -- out.png a.glb [b.glb ...]

Each GLB is imported with the add-on's own importer conventions (glTF +Z -> Blender -Y) and
rendered from the front and from a 3/4 angle side by side, for a quick visual check.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
out, files = Path(argv[0]).resolve(), [Path(a).resolve() for a in argv[1:]]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "VERTEX"
scene.render.resolution_x, scene.render.resolution_y = 512, 512
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("w")
scene.world.color = (0.05, 0.05, 0.08)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"

tiles = []
for i, path in enumerate(files):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    obj = objs[0]
    base = obj.matrix_world.copy()  # the glTF importer uses quaternion rotation: turn via matrices
    for a, angle in enumerate((0.0, 35.0)):
        obj.matrix_world = Matrix.Rotation(math.radians(angle), 4, "Z") @ base
        bpy.context.view_layer.update()
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        center = sum(corners, Vector()) / 8
        size = max((max(c[k] for c in corners) - min(c[k] for c in corners)) for k in range(3))
        cam.data.ortho_scale = size * 1.15
        cam.location = center + Vector((0.0, -10.0, 0.0))
        cam.rotation_euler = (math.radians(90), 0.0, 0.0)  # looks along +Y: the Front view
        tile = out.with_name(f"{out.stem}_{i}_{a}.png")
        scene.render.filepath = str(tile)
        bpy.ops.render.render(write_still=True)
        tiles.append(tile)
    obj.hide_render = True

# stitch the tiles into one picture: one row per file, front | 3/4
import numpy as np  # noqa: E402

imgs = [bpy.data.images.load(str(t)) for t in tiles]
w, h = imgs[0].size
rows = len(files)
canvas = np.zeros((rows * h, 2 * w, 4), np.float32)
for k, img in enumerate(imgs):
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    r, c = divmod(k, 2)
    canvas[(rows - 1 - r) * h : (rows - r) * h, c * w : (c + 1) * w] = px.reshape(h, w, 4)
sheet = bpy.data.images.new("sheet", 2 * w, rows * h)
sheet.pixels.foreach_set(canvas.ravel())
sheet.filepath_raw = str(out)
sheet.file_format = "PNG"
sheet.save()
for t in tiles:
    t.unlink()
print("saved", out)
