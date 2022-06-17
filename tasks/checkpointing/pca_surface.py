
import torch

from utils import sinebow

from .surface import SurfaceCheckpointer

class PCASurfaceCheckpointer(SurfaceCheckpointer):

    def run(self, model, experiment, ckpt_info):
        super().run(model, experiment, ckpt_info)

        # generate html file for visualization of patches
        if self.save_patches:
            prefix_name = self.build_prefix_name(['patches'], None)
            self.create_visualization_page('patch', ckpt_info.checkpoint_dir, prefix=prefix_name)


    def checkpoint_sample(self, sample, model, experiment, ckpt_info):

        ##### get data out of the dictionary
        num_patches  = sample['num_patches']
        faces        = sample['faces']
        param_idxs   = sample['param_idxs']
        visual_W     = sample['visual_W']
        global_faces = sample['global_faces']
        global_gt    = sample['global_gt']
        shape_name   = sample['name']
        global_gt    = self.move_to_device(global_gt)

        N = global_gt.size(0)

        ##### create book-keeping variables
        global_points      = torch.zeros((N, 3), device=global_gt.device)
        global_count       = torch.zeros((N, 1), device=global_gt.device)
        vert_colors        = torch.zeros_like(global_gt).cpu()
        global_distortions = torch.zeros((N, 4)) if ckpt_info.generate_report else None

        ##### cycle through all the patches
        for patch_idx in range(num_patches):

            ##### checkpoint a patch (generate and save to disk)
            pred_points, patch_distortion, patch_colors = self.checkpoint_patch(model, sample, patch_idx, faces[patch_idx], shape_name, ckpt_info)

            ##### get info to assemble patch onto a single mesh
            W         = self.move_to_device(visual_W[patch_idx])
            param_idx = self.move_to_device(param_idxs[patch_idx])

            ##### assemble patch over the original mesh
            global_points[param_idx] += pred_points[0, :param_idx.size(0)] * W.reshape(-1, 1) # indexed 0 to remove batch size
            global_count[param_idx]  += W.reshape(-1, 1) # 1.0
            vert_colors[param_idx] = patch_colors # assemble colors

            if patch_distortion is not None:
                self.replace_distortion_measures(global_distortions, param_idx, patch_distortion)

        ##### finish points interpolation based on their weights
        interpolated_points = global_points / global_count
        pt_distance         = (global_gt - interpolated_points).pow(2).sum(-1)

        ##### add distortion measures to the dictionary to save them with the surface
        scalars = {}
        if ckpt_info.generate_report:
            scalars = self.distortion_to_dict(global_distortions)
        scalars['l2error'] = pt_distance

        ##### save full surface and model
        self.append_surface_data(interpolated_points, global_faces, None, ['full', shape_name], ckpt_info, scalars=scalars)
        self.append_surface_data(global_gt, global_faces, None, ['GT', 'full', shape_name], ckpt_info, constant=True)

        ##### produce final html report
        if ckpt_info.generate_report:
            self.report_error_metrics(global_gt, interpolated_points, global_faces)


    def checkpoint_patch(self, model, sample, patch_idx, faces, shape_name, ckpt_info):
        num_patches = sample['num_patches']

        pred_points, pt_distance = self.forward_checkpoint(model, sample, patch_idx)

        ##### prepare patch colors
        patch_colors    = torch.zeros_like(pred_points).cpu()
        patch_colors[:] = torch.from_numpy(sinebow(patch_idx / num_patches)).float() * 255.0

        ##### save patch if needed
        patch_distortion = self.patch_report(pred_points, pt_distance, patch_idx, sample, ckpt_info)

        ## save surface of a patch with scalar values
        if self.save_patches:
            patch_scalars = {}
            if ckpt_info.generate_report:
                patch_scalars = self.distortion_to_dict(patch_distortion)
            patch_scalars['l2error'] = pt_distance

            self.append_surface_data(pred_points, faces, None, ['patch', shape_name, f'{patch_idx:0>3}'], ckpt_info, scalars=patch_scalars, colors=patch_colors)
            self.save_gt_patch(sample, patch_distortion[:, :2], patch_idx, ckpt_info)

        return pred_points, patch_distortion, patch_colors


    def forward_checkpoint(self, model, sample, patch_idx):
        ## forward pass for a single patch
        params = sample['param'][patch_idx]
        gts    = sample['gts'][patch_idx]
        idxs   = sample['idxs'][patch_idx]
        Rs     = sample['Rs'][patch_idx]
        ts     = sample['ts'][patch_idx]
        Cs     = sample['Cs'][patch_idx]

        param = self.move_to_device(params)
        idx   = self.move_to_device(idxs)
        gt    = self.move_to_device(gts)
        R     = self.move_to_device(Rs)
        t     = self.move_to_device(ts)

        points = model(param.unsqueeze(0), idx) / Cs

        pt_distance = (gt - points).pow(2).sum(-1)

        points = points.matmul(R.t()) + t

        return points, pt_distance


    def save_gt_patch(self, sample, distortion, patch_idx, ckpt_info):

        name   = sample['name']
        gts    = sample['gts'][patch_idx]
        faces  = sample['faces'][patch_idx]
        params = sample['param'][patch_idx]
        Cs     = sample['Cs'][patch_idx]
        Rs     = sample['Rs'][patch_idx]
        ts     = sample['ts'][patch_idx]

        pts = (gts / Cs).matmul(Rs.t()) + ts

        patch_scalars = self.distortion_to_dict(distortion.repeat(1,2))
        del patch_scalars['scale']
        del patch_scalars['angle']

        self.append_surface_data(pts, faces, params, ['GT', 'patch', name, f'{patch_idx:0>3}'], ckpt_info, scalars=patch_scalars, constant=True)


    def patch_report(self, pred_points, pt_distance, patch_idx, sample, ckpt_info):
        ## add data to html report for a patch
        distortion = None
        if ckpt_info.generate_report:
            distortion = self.compute_patch_distortion(pred_points, sample, patch_idx)

            self.html_logs.add_entry('reconstruction', pt_distance, f'Patch {patch_idx}')
            self.report_surface_distortion(distortion, f'Patch {patch_idx}')

        return distortion


    def compute_patch_distortion(self, points, sample, patch_idx):
        params = sample['param'][patch_idx]
        gts    = sample['gts'][patch_idx]
        faces  = sample['faces'][patch_idx]

        param = self.move_to_device(params)
        gt    = self.move_to_device(gts)
        face  = self.move_to_device(faces)

        distortion = self.get_distortion_tensor(gt, points, param, face)

        return distortion


    def replace_distortion_measures(self, global_distortions, param_idxs, patch_distortion):
        ## assemble distortion measures into global distortion
        global_distortions[param_idxs, 0] = torch.max(patch_distortion[:,0], global_distortions[param_idxs, 0])
        global_distortions[param_idxs, 1] = torch.max(patch_distortion[:,1], global_distortions[param_idxs, 1])
        global_distortions[param_idxs, 2] = torch.max(patch_distortion[:,2], global_distortions[param_idxs, 2])
        global_distortions[param_idxs, 3] = torch.max(patch_distortion[:,3], global_distortions[param_idxs, 3])
