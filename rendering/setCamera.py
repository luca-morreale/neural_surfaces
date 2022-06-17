
import bpy
import numpy as np


def setCamera(camLocation, lookAt_rot = (0,0,0), focalLength = 35):

    la_x = lookAt_rot[0] * np.pi / 180.0
    la_y = lookAt_rot[1] * np.pi / 180.0
    la_z = lookAt_rot[2] * np.pi / 180.0

    # initialize camera
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add(location = camLocation) # name 'Camera'
    else:
        bpy.data.objects['Camera'].location = camLocation

    cam = bpy.context.object
    cam.data.lens = focalLength
    # loc = mathutils.Vector(lookAtLocation)
    # lookAt(cam, loc)
    cam.rotation_euler = (la_x, la_y, la_z)
    return cam


def set_camera_ortho(camscale):
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            obj.data.type = 'ORTHO'
            obj.data.ortho_scale = camscale
