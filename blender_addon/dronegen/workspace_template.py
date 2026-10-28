"""Runs inside a background Blender (``blender --background --factory-startup --python this -- out.blend``).

Builds a file that holds only the «DroneGen» workspace, made from Blender's own «Texture Paint»
layout: pictures (Image Editor) on the left, the 3D view on the right, the outliner and the
properties editor above and below each other on the far right. The add-on appends it with
``workspace.append_activate`` into the user's session. Built on demand, with the user's Blender,
so it always matches that Blender's version.
"""

import sys

import bpy

NAME = "DroneGen"
SOURCE = "Texture Paint"


def build(path: str) -> None:
    ws = bpy.data.workspaces[SOURCE]
    ws.name = NAME
    ws.object_mode = "OBJECT"  # the paint mode of the source layout would change the user's active object
    # The other workspaces stay in the file: appending takes the one by name, and removing
    # workspaces that a window shows crashes Blender.
    for screen in ws.screens:
        for area in screen.areas:
            space = area.spaces.active
            if area.type == "IMAGE_EDITOR":
                space.ui_mode = "VIEW"  # not paint tools
                space.show_region_toolbar = False
            elif area.type == "VIEW_3D":
                space.shading.type = "SOLID"
                space.shading.color_type = "VERTEX"  # the figure's colours
                space.show_region_ui = True  # the N panel with the DroneGen tab
                space.show_region_toolbar = False
            elif area.type == "OUTLINER":
                space.show_restrict_column_viewport = True
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)


if __name__ == "__main__":
    build(sys.argv[sys.argv.index("--") + 1])
