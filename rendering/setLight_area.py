
import bpy
import numpy as np


def setLight_area(location, rotation, intensity, size=1.3, shadow=True, color=None):

    euler_x = rotation[0] * np.pi / 180.0
    euler_y = rotation[1] * np.pi / 180.0
    euler_z = rotation[2] * np.pi / 180.0

    prev_lights = bpy.data.lights.keys()

    bpy.ops.object.light_add(type='AREA', radius=size, \
                            location=location, rotation=(euler_x, euler_y, euler_z))

    new_light_name = list(set(bpy.data.lights.keys()) - set(prev_lights))[0]

    areaL = bpy.data.lights[new_light_name]
    areaL.energy = intensity
    areaL.shape  = "DISK"

    if color is not None:
        areaL.color = color

    areaL.cycles.cast_shadow = shadow

    return areaL
