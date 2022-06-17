
import torch

from runners import CheckpointRunner


class SurfaceMapCheckpointer(CheckpointRunner):

    def checkpoint_sample(self, sample, model, experiment, ckpt_info):

        source_uvs      = self.move_to_device(sample['source_points'])
        R               = self.move_to_device(sample['R'])
        t               = self.move_to_device(sample['t'])
        source_C        = sample['C_source']
        target_C        = sample['C_target']
        lands_source    = self.move_to_device(sample['landmarks'])
        source_boundary = self.move_to_device(sample['boundary'])

        target_domain   = sample['target_domain']

        if target_domain is not None:
            target_domain = self.move_to_device(target_domain)

        if 'map_gt' in sample:
            points3D_gt     = self.move_to_device(sample['target_points_3D'])
            map             = self.move_to_device(sample['map_gt'])
            map_mask        = self.move_to_device(sample['map_mask'])


        torch.set_grad_enabled(True)
        source_uvs.requires_grad_(True)

        points3D_target, points2D_target, points3D_source = model(source_uvs, [R, t])
        # points3D_target, points2D_target, points3D_source = model(source_uvs, None)
        points3D_target *= target_C
        points3D_source *= source_C

        ###### LOSSES
        domain_mask              = experiment['loss'].domain.domain_mask(points2D_target, target_domain)
        loss_point_distortion, _ = experiment['loss'].surf_map_loss.compute_per_point_distortion(points3D_target, points2D_target, source_uvs, points3D_source)
        loss_point_distortion[~domain_mask] = 1.0e10
        torch.set_grad_enabled(False)

        loss_euclidean = torch.zeros(points3D_target.size(0), device=points3D_target.device).float() + 1.0e10
        if 'map_gt' in sample:
            loss_euclidean[map_mask] = (points3D_target[map_mask] - points3D_gt[map]).pow(2).sum(-1)

        losses = {'euclidean': loss_euclidean.detach(), 'distortion':loss_point_distortion.detach()}

        source_boundary_mapped = model.forward_map(source_boundary, [R, t])
        lands_source_mapped    = model.forward_map(lands_source, [R, t])

        ###### SAVE images of 2D domain
        self.save_image_visual(sample, lands_source_mapped, points2D_target, ckpt_info)

        ###### SAVE surfaces
        self.save_all_surfaces(sample, points3D_target, points2D_target, points3D_source, losses, model, experiment, ckpt_info)

        if ckpt_info.generate_report:
            self.report_error_metrics(experiment, sample, points2D_target, source_boundary_mapped, lands_source_mapped, domain_mask, losses)


    def save_all_surfaces(self, sample, points3D_target, points2D_target, points3D_source, losses, model, experiment, ckpt_info):
        source_uvs    = sample['source_points']
        target_uvs    = sample['target_points']
        source_faces  = sample['source_faces']
        target_faces  = sample['target_faces']
        target_name   = sample['target_name']
        source_name   = sample['source_name']
        target_domain = sample['target_domain']
        R             = sample['R']
        t             = sample['t']
        visual_uvs    = sample['source_points'] if 'visual_uv' not in sample else sample['visual_uv']

        ### filter overlapping mesh
        domain_mask = experiment['loss'].domain.domain_mask(points2D_target, target_domain).cpu()
        faces_mask = domain_mask[source_faces].prod(dim=-1).bool()
        faces_filtered = source_faces[faces_mask]

        ## save mapped surface (source -> target)
        if type(visual_uvs) is dict:
            for k, v in visual_uvs.items():
                self.append_surface_data(points3D_source, source_faces, v,   [source_name, k], ckpt_info, constant=True)
                self.append_surface_data(points3D_target, faces_filtered, v, [target_name, k, 'filtered'], ckpt_info, scalars=losses)
        # replace uvs with XY coordinates on the source mesh
        for selection, name in zip([[0,1], [0,2], [1,2]], ['sxy', 'sxz', 'syz']):
            self.append_surface_data(points3D_source, source_faces, points3D_source[:, selection], [source_name, name], ckpt_info, constant=True)
            self.append_surface_data(points3D_target, faces_filtered, points3D_source[:, selection], [target_name, name, 'filtered'], ckpt_info, scalars=losses)
        self.append_surface_data(points3D_target, faces_filtered, source_uvs, [target_name, 'filtered'], ckpt_info, scalars=losses)

        ## save original source surface (constant)
        self.append_surface_data(points3D_source, source_faces, source_uvs, [source_name], ckpt_info, constant=True)

        ## save original target surface (constant)
        target_uvs      = self.move_to_device(target_uvs)
        original_target = model.target_surface(target_uvs)
        self.append_surface_data(target_points_3D, target_faces, target_uvs, [target_name, 'original'], ckpt_info, constant=True)
        for selection, name in zip([[0,1], [0,2], [1,2]], ['sxy', 'sxz', 'syz']):
            self.append_surface_data(original_target, target_faces, original_target[:, selection], [target_name, 'original', name], ckpt_info, constant=True)
        self.append_surface_data(original_target, target_faces, target_uvs, [target_name, 'original'], ckpt_info, constant=True)

        target_points_3D = sample['target_points_3D']
        target_uvs       = sample['target_points'] if 'visual_uv_target' not in sample else sample['visual_uv_target']
        if type(target_uvs) is dict:
            for k, v in target_uvs.items():
                self.append_surface_data(target_points_3D, target_faces, v, [target_name, k, 'original'], ckpt_info, constant=True)

        ##### oversampled domain surface
        if 'oversampled_param' in sample:
            source_uvs   = self.move_to_device(sample['oversampled_param'])
            source_faces = self.move_to_device(sample['oversampled_faces'])
            points3D_target, points2D_target, _ = model(source_uvs, [R, t])
            # points3D_target, points2D_target, _ = model(source_uvs, None)

            ### filter overlapping mesh
            domain_mask = experiment['loss'].domain.domain_mask(points2D_target, target_domain).cpu()
            faces_mask = domain_mask[source_faces].prod(dim=-1).bool()
            faces_filtered = source_faces[faces_mask]

            self.append_surface_data(points3D_target, faces_filtered, source_uvs, [target_name, 'filtered', 'oversampled'], ckpt_info)

    def save_image_visual(self, sample, lands_source_mapped, points2D_target, ckpt_info):
        source_uvs      = sample['source_points']
        target_uvs      = sample['target_points']
        source_faces    = sample['source_faces']
        target_faces    = sample['target_faces']
        target_name     = sample['target_name']
        source_name     = sample['source_name']
        lands_target    = sample['target_landmarks']

        prefix_name = self.build_prefix_name([source_name], None)
        self.save_uvmesh_image(source_uvs, source_faces, ckpt_info.checkpoint_dir, prefix=prefix_name)
        prefix_name = self.build_prefix_name([target_name], None)
        self.save_uvmesh_image(target_uvs, target_faces, ckpt_info.checkpoint_dir, prefix=prefix_name)
        self.save_landmarks_image(target_uvs, target_faces, [lands_target, lands_source_mapped], ckpt_info.checkpoint_dir, prefix=prefix_name)
        prefix_name = self.build_prefix_name([source_name, target_name], None)
        self.save_uvmesh_image(points2D_target, source_faces, ckpt_info.checkpoint_dir, prefix=prefix_name)
        prefix_name = self.build_prefix_name([source_name, target_name], ckpt_info)
        self.save_uvmesh_overlap([points2D_target, target_uvs], [source_faces, target_faces], ckpt_info.checkpoint_dir, prefix=prefix_name)


    def create_report_data_obj(self):
        super().create_report_data_obj(self)
        self.html_logs.add_log('error', "Map Errors")


    def report_error_metrics(self, experiment, sample, points2D_target, source_boundary_mapped, lands_source_mapped, domain_mask, losses):
        lands_target    = self.move_to_device(sample['target_landmarks'])
        source_boundary = self.move_to_device(sample['boundary'])

        loss_domain    = points2D_target[~domain_mask].pow(2).sum(-1)
        loss_landmarks = (lands_target - lands_source_mapped).pow(2).sum(-1)
        loss_boundary, _ = experiment['loss'].boundary_loss(source_boundary_mapped, source_boundary)

        self.html_logs.add_entry('error', loss_landmarks.sqrt().squeeze(), 'Landmarks')
        self.html_logs.add_entry('error', loss_boundary.sqrt().squeeze(), 'Boundary error')
        self.html_logs.add_entry('error', loss_domain.sqrt().squeeze(), 'Domain error')
        for k, v in losses.items():
            self.html_logs.add_entry('error', v.squeeze(), k)
