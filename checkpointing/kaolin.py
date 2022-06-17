
import os

from .mixin import Mixin


class KaolinCheckpointing(Mixin):

    timelapse=None
    save_kaolin=True

    def check_imports(self):
        ### check if library exists, if does not remove function
        try:
            from kaolin.visualize import Timelapse
        except ImportError as err:
            print('Error missing library ' + str(err))
            self.timelapse_mesh = self.empty_function


    def timelapse_mesh(self, points3D, faces, uvs, tags, ckpt_info, constant=False):
        ### Save mesh w/ kaolin

        if not self.save_kaolin:
            return

        from kaolin.visualize import Timelapse

        # if (not constant) or (constant and ckpt_info.save_constant_data):
        if constant and not ckpt_info.save_constant_data:
            return

        name = self.build_prefix_name(tags, None, trail=False)

        if self.timelapse is None:
            self.timelapse = Timelapse(os.path.join(ckpt_info.checkpoint_dir, 'timelapse'))
            self.epochs = {}


        it = self.epochs.get(name, 0)
        self.epochs[name] = it + 1

        if it != 0:
            faces = None
        else:
            faces = faces.contiguous()

        self.timelapse.add_mesh_batch(
                category=name,
                iteration=it,
                faces_list=[faces],
                uvs_list=[uvs],
                vertices_list=[points3D.detach().contiguous()])
