
import torch

from .pca_surface import PCASurfaceCheckpointer

class ConvSurfaceCheckpointer(PCASurfaceCheckpointer):


    def forward_checkpoint(self, model, sample, patch_idx):
        params       = sample['param'][patch_idx]
        gts          = sample['gts'][patch_idx]
        param_idxs   = sample['param_idxs'][patch_idx]
        idxs         = sample['idxs'][patch_idx]
        coarse_param = sample['coarse_param']

        param = self.move_to_device(params)
        idx   = self.move_to_device(idxs)
        gt    = self.move_to_device(gts)
        coarse_param_ = self.move_to_device(coarse_param[param_idxs])

        torch.set_grad_enabled(True)
        coarse_param_.requires_grad_(True)

        points = model(param.unsqueeze(0), idx, coarse_param_.unsqueeze(0))
        torch.set_grad_enabled(False)

        pt_distance = (gt - points).pow(2).sum(-1)

        return points, pt_distance


    def checkpoint_sample(self, sample, model, experiment, ckpt_info):
        super().checkpoint_sample(sample, model, experiment, ckpt_info)

        global_faces = sample['global_faces']
        coarse_param = sample['coarse_param']
        shape_name   = sample['name']
        coarse_param_ = self.move_to_device(coarse_param)

        points_uv = model.mlp_coarse(coarse_param_.unsqueeze(0))

        self.append_surface_data(points_uv.squeeze(), global_faces, None, ['full', 'uvs', shape_name], ckpt_info)
