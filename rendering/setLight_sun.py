
import bpy
import numpy as np


def setLight_sun(rotation_euler, strength, shadow_soft_size = 0.05, shadow=True):
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x,y,z)

    prev_lights = bpy.data.lights.keys()

    if 'Sun' not in bpy.data.lights:
       bpy.ops.object.light_add(type = 'SUN', rotation = angle)
    else:
        bpy.data.objects['Sun'].rotation_euler = angle

    new_light_name = list(set(bpy.data.lights.keys()) - set(prev_lights))[0]

    lamp = bpy.data.lights[new_light_name]
    lamp.use_nodes = True
    # lamp.shadow_soft_size = shadow_soft_size # this is for older blender 2.8
    lamp.angle = shadow_soft_size

    lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength

    lamp.cycles.cast_shadow = shadow

    return lamp
