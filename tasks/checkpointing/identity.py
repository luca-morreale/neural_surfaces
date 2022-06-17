
import torch

from checkpointing import AllCheckpointing
from runners import CheckpointRunner


class IdentityCheckpointer(CheckpointRunner, AllCheckpointing):

    def checkpoint_sample(self, sample, model, experiment, ckpt_info):
        source_uvs   = sample['source_points']
        source_faces = sample['source_faces']

        source_uvs = self.move_to_device(source_uvs)

        torch.set_grad_enabled(True)
        source_uvs.requires_grad_(True)

        ## forward
        _, points2D_target, _ = model(source_uvs)
        torch.set_grad_enabled(False)

        ## save output as 2D mesh layout
        prefix_name = self.build_prefix_name(['source'], ckpt_info)
        self.save_uvmesh_image(source_uvs, source_faces, ckpt_info.checkpoint_dir, prefix=prefix_name)
        prefix_name = self.build_prefix_name(['source', 'target'], ckpt_info)
        self.save_uvmesh_overlap([points2D_target, source_uvs], [source_faces, source_faces], ckpt_info.checkpoint_dir, prefix=prefix_name)


    def end_checkpointing(self, model, name, ckpt_info):
        super().end_checkpointing(self, model, name, ckpt_info)
        ## save model neural map
        self.save_model(ckpt_info.checkpoint_dir, model.neural_map, name='_neural_map')