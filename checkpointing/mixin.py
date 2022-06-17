
import os

from utils import mkdir
from utils import tensor_to_numpy


class Mixin(object):

    def empty_function(self, **kwargs):
        pass

    # =================== Move torch data to numpy ===================== #
    def move_to_numpy(self, data):
        return tensor_to_numpy(data, squeeze=True)

    # =================== Create folder ================================ #
    def compose_out_folder(self, checkpoint_dir, sub_folders):
        out_folder = os.path.join(checkpoint_dir, *sub_folders)
        mkdir(out_folder)

        return out_folder

    # =================== Concat tags ================================ #
    def build_prefix_name(self, tags, ckpt_info, trailing=True):
        ## join list of tags
        out = '_'.join(tags) + '_'
        if self.save_timelapse and ckpt_info is not None:
            out +=  '{0:05d}_'.format(ckpt_info.epoch)

        if not trailing:
            out = out[:-1]
        return out
