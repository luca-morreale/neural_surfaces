
import bpy


def setLight_point(location, intensity, radius=0.25, shadow=True):

    prev_lights = bpy.data.lights.keys()

    bpy.ops.object.light_add(type='POINT', radius=radius, location=location)

    new_light_name = list(set(bpy.data.lights.keys()) - set(prev_lights))[0]

    pointL = bpy.data.lights[new_light_name]
    pointL.energy = intensity

    pointL.cycles.cast_shadow = shadow

    return pointL
