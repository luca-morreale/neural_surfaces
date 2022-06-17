
import bpy


def setClamp_samples(clamp_direct=1.0, clamp_indirect=1.0):

    bpy.context.scene.cycles.sample_clamp_indirect = clamp_indirect
    bpy.context.scene.cycles.sample_clamp_direct = clamp_direct
