
import logging
import numpy as np
import os
import sys
from pathlib import Path

from .mixin import Mixin


class BlenderCheckpointing(Mixin):

    def check_imports(self):
        ### check if library exists, if does not remove function
        try:
            import bpy
        except ImportError as err:
            print('Error missing library ' + str(err))
            self.render = self.empty_function


    def render(self):
        ### Render files based on configuration

        import bpy
        import rendering

        config_rendering_path = self.config['rendering']['config']

        if not os.path.exists(config_rendering_path):
            return
        if not os.path.isfile(config_rendering_path):
            return

        rendering_params = rendering.parse_configuration(config_rendering_path)

        mesh_folder = self.compose_out_folder(self.checkpoint_dir, ['mesh'])
        out_folder = self.compose_out_folder(self.checkpoint_dir, ['render'])


        base_path_mesh = Path(mesh_folder)

        ### suppress cmd output
        logging.warning('WARNING: suppression ON, do not kill')
        logfile = 'blender_render.log'
        open(logfile, 'w').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)


        for scene_id, scene in enumerate(rendering_params):

            for mesh_path in list(base_path_mesh.glob(scene['file_pattern'])):

                file_name = mesh_path.stem

                rendering.blenderInit(scene['image']['res_x'], scene['image']['res_y'], scene['image']['samples'], scene['image']['exposure'])
                bpy.data.scenes['Scene'].view_settings.view_transform = 'Standard'

                location = scene['mesh']['location']
                rotation = scene['mesh']['rotation']
                scale    = scene['mesh']['scale']

                if mesh_path.name[-3:] == 'obj':
                    mesh = rendering.readOBJ(str(mesh_path), location, rotation, scale)
                elif mesh_path.name[-3:] == 'ply':
                    mesh = rendering.readPLY(str(mesh_path), location, rotation, scale)
                else:
                    continue

                ## set shading
                bpy.ops.object.shade_flat()
                if scene['shading'] == 'smooth':
                    bpy.ops.object.shade_smooth()


                ## set material
                if ('use_vertex_colors', True) in scene['mesh'].items():
                    meshVColor = rendering.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
                    rendering.setMat_VColor(mesh, meshVColor)
                elif ('use_texture', True) in scene['mesh'].items():
                    meshColor = rendering.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
                    texturePath   = scene['mesh']['texture']
                    uvscale       = scene['mesh']['uv_scale'] if 'uv_scale' in scene['mesh'] else 1.0

                    uvrotation    = scene['mesh']['uv_rotation'] if 'uv_rotation' in scene['mesh'] else 0.0
                    if type(uvrotation) == list:
                        uvrotation = [ el * np.pi/180.0 for el in uvrotation ]
                    else:
                        uvrotation *= np.pi/180.0

                    uvtranslation = scene['mesh']['uv_translation'] if 'uv_translation' in scene['mesh'] else [0.0, 0.0]

                    # using relative path gives us weired bug...
                    rendering.setMat_texture(mesh, texturePath, meshColor, scale=uvscale, \
                                    rotation=uvrotation, translation=uvtranslation)
                else:
                    meshColor  = rendering.colorObj(scene['mesh']['color'], 0.5, 1.0, 1.0, 0.0, 2.0)
                    AOStrength = scene['AOStrength']
                    rendering.setMat_singleColor(mesh, meshColor, AOStrength)

                ## set invisible plane (shadow catcher)
                groundCenter    = (0,0,0)
                shadowDarkeness = 0.7
                groundSize      = 20
                rendering.invisibleGround(groundCenter, groundSize, shadowDarkeness)

                for light in scene['lights']:

                    if light['type'] == 'area':
                        color = light['color'] if 'color' in light else None
                        rendering.setLight_area(light['location'], light['rotation'], light['intensity'], size=light['size'], shadow=light['shadow'], color=color)
                    elif light['type'] == 'sun':
                        rendering.setLight_sun(light['rotation'], light['intensity'], shadow=light['shadow'])

                # rendering.setClamp_samples(clamp_indirect=1.0)

                ## set ambient light
                ambientColor = (0.0, 0.0, 0.0, 1)
                rendering.setLight_ambient(ambientColor)


                focalLength = 45
                cam = rendering.setCamera(scene['camera']['location'], scene['camera']['lookAt'], focalLength)

                # set_camera_ortho(args)
                rendering.set_camera_ortho(scene['camera']['scale'])


                outputPath = '{}/{}_{}'.format(out_folder, file_name, scene_id)
                ## save blender file
                bpy.ops.wm.save_mainfile(filepath=outputPath + '.blend')

                ## save rendering

                rendering.renderImage(outputPath + '.png', cam)

                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()


        ### undo suppression
        os.close(1)
        os.dup(old)
        os.close(old)
        logging.warning('WARNING: suppression OFF')
