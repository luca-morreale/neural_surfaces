
import bpy
import numpy as np


def readPLY(filePath, location, rotation_euler, scale):
    # example input types:
    # - location = (0.5, -0.5, 0)
    # - rotation_euler = (90, 0, 0)
    # - scale = (1,1,1)
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x,y,z)
    prev = []
    for ii in range(len(list(bpy.data.objects))):
        prev.append(bpy.data.objects[ii].name)
    bpy.ops.import_mesh.ply(filepath=filePath)
    after = []
    for ii in range(len(list(bpy.data.objects))):
        after.append(bpy.data.objects[ii].name)
    name = list(set(after) - set(prev))[0]

    mesh = bpy.data.objects[name]

    # mesh.location = location
    mesh.rotation_euler = angle
    mesh.scale = scale
    bpy.context.view_layer.update()

    obj = bpy.context.object
    # get the minimum z-value of all vertices after converting to global transform
    lowest_pt = min([(obj.matrix_world @ v.co).z for v in obj.data.vertices])
    avg_x = np.average([(obj.matrix_world @ v.co).x for v in obj.data.vertices])
    avg_y = np.average([(obj.matrix_world @ v.co).y for v in obj.data.vertices])

    # transform the object
    obj.location.z -= lowest_pt
    obj.location.x -= avg_x
    obj.location.y -= avg_y

    obj.location.x += location[0]
    obj.location.y += location[1]
    obj.location.z += location[2]
    bpy.context.view_layer.update()

    return mesh