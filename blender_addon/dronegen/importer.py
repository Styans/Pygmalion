"""A GLB from the server -> one mesh object in the "DroneGen" collection, centred and sized.

The server writes glTF axes (+Y up, picture facing +Z); Blender's importer turns that into
Z up with the picture facing the Front view (-Y). Transforms are baked into the mesh, so
the object sits at the origin with scale 1 and its largest side equals the chosen size.
Vertex colours get a material that uses them, so the viewport (Material Preview) and
point samplers such as Pointille see the colours.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

COLLECTION = "DroneGen"
MATERIAL = "DroneGen Vertex Color"


def collection(scene) -> bpy.types.Collection:
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if scene.collection.children.get(coll.name) is None:
        scene.collection.children.link(coll)
    return coll


def _gltf_import(path: Path) -> None:
    kwargs = {"filepath": str(path)}
    props = bpy.ops.import_scene.gltf.get_rna_type().properties
    if "import_scene_as_collection" in props:
        kwargs["import_scene_as_collection"] = False
    wm = bpy.context.window_manager
    window = wm.windows[0] if wm is not None and wm.windows else None
    if window is not None and bpy.context.window is None:  # called from a timer
        with bpy.context.temp_override(window=window):
            bpy.ops.import_scene.gltf(**kwargs)
    else:
        bpy.ops.import_scene.gltf(**kwargs)


def normalize(me: bpy.types.Mesh, size: float) -> None:
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(axis=0), co.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise ValueError("пустая сетка")
    s = size / extent
    me.transform(Matrix.Diagonal((s, s, s, 1.0)) @ Matrix.Translation(-(lo + hi) / 2.0))
    me.update()


def vertex_color_material(attr_name: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(MATERIAL)
    if mat is None:
        mat = bpy.data.materials.new(MATERIAL)
        mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next((n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None) or nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    vc = next((n for n in nodes if n.bl_idname == "ShaderNodeVertexColor"), None)
    if vc is None:
        vc = nodes.new("ShaderNodeVertexColor")
        vc.location = (bsdf.location.x - 300, bsdf.location.y)
    vc.layer_name = attr_name
    links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def import_glb(context, path: Path, name: str, size: float, info: dict) -> bpy.types.Object:
    before = {o.name for o in bpy.data.objects}
    _gltf_import(path)
    new = [o.name for o in bpy.data.objects if o.name not in before]
    meshes = [bpy.data.objects[n] for n in new if bpy.data.objects[n].type == "MESH"]
    if not meshes:
        for n in new:
            bpy.data.objects.remove(bpy.data.objects[n])
        raise RuntimeError("в полученном файле нет сетки")
    obj = meshes[0]
    if len(meshes) > 1:  # our meshes are one piece; merge anything unexpected into it
        with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=meshes, selected_editable_objects=meshes):
            bpy.ops.object.join()
    world = obj.matrix_world.copy()
    obj.parent = None
    obj.data.transform(world)
    obj.matrix_world = Matrix.Identity(4)
    for n in new:  # helper empties of the glTF scene
        o = bpy.data.objects.get(n)
        if o is not None and o is not obj:
            bpy.data.objects.remove(o)
    normalize(obj.data, size)

    coll = collection(context.scene)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    obj.name = name
    obj.data.name = name

    me = obj.data
    if len(me.color_attributes):
        attr = me.color_attributes.active_color or me.color_attributes[0]
        me.color_attributes.active_color = attr
        me.color_attributes.render_color_index = list(me.color_attributes).index(attr)
        me.materials.clear()
        me.materials.append(vertex_color_material(attr.name))
    for key, value in info.items():
        obj[f"dronegen_{key}"] = value

    view_layer = context.view_layer
    for o in view_layer.objects:
        if o.select_get():
            o.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj
    view_layer.update()  # bounding box and dimensions of the rewritten mesh
    return obj
