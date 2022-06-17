
import bpy
import os


def setMat_texture(mesh, texturePath, meshColor, scale=1.0, rotation=0.0, translation=[0,0]):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # set principled BSDF
    PRI = tree.nodes["Principled BSDF"]
    PRI.inputs['Roughness'].default_value = 1.0
    PRI.inputs['Sheen Tint'].default_value = 0

    TI = tree.nodes.new('ShaderNodeTexImage')
    absTexturePath = os.path.abspath(texturePath)
    TI.image = bpy.data.images.load(absTexturePath)

    # set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Saturation'].default_value = meshColor.S
    HSVNode.inputs['Value'].default_value = meshColor.V
    HSVNode.inputs['Hue'].default_value = meshColor.H

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshColor.B
    BCNode.inputs['Contrast'].default_value = meshColor.C

    tree.links.new(TI.outputs['Color'], HSVNode.inputs['Color'])
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], PRI.inputs['Base Color'])


    MappingNode = tree.nodes.new('ShaderNodeMapping')
    TCNode = tree.nodes.new('ShaderNodeTexCoord')


    MappingNode.inputs['Scale'].default_value[0] = scale
    MappingNode.inputs['Scale'].default_value[1] = scale
    MappingNode.inputs['Scale'].default_value[2] = scale

    if type(rotation) == list:
        for i in range(len(rotation)):
            MappingNode.inputs['Rotation'].default_value[i] = rotation[i]
    else:
        MappingNode.inputs['Rotation'].default_value[2] = rotation

    for i in range(len(translation)):
        MappingNode.inputs['Location'].default_value[i] = translation[i]

    tree.links.new(MappingNode.outputs['Vector'], TI.inputs['Vector'])
    tree.links.new(TCNode.outputs['UV'], MappingNode.inputs['Vector'])
