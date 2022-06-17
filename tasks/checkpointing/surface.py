
import logging
import torch

from runners import CheckpointRunner


class SurfaceCheckpointer(CheckpointRunner):

    def checkpoint_sample(self, sample, model, experiment, ckpt_info):
        param = sample['param']
        gts   = sample['gts']
        faces = sample['faces']
        name  = sample['name']
        param = self.move_to_device(param)
        gts   = self.move_to_device(gts)
        faces = self.move_to_device(faces)

        pred_points = model(param)
        pt_distance = (pred_points - gts).pow(2).sum(-1)

        scalars = {}
        if ckpt_info.generate_report:
            distortion = self.get_distortion_tensor(gts, pred_points, param, faces)
            scalars    = self.distortion_to_dict(distortion)
            self.report_surface_distortion(distortion, f'{name}')
        scalars['l2error'] = pt_distance

        self.append_surface_data(pred_points, faces, param, [name], ckpt_info, scalars=scalars)
        self.append_surface_data(gts, faces, param, ['GT', name], ckpt_info, constant=True)

        ## check using sampled param
        if 'oversampled_param' in sample:
            param_large = sample['oversampled_param']
            faces_large = sample['oversampled_faces']
            param_large = self.move_to_device(param_large)
            pred_points_large = model(param_large)

            self.append_surface_data(pred_points_large, faces_large, param_large, [name, 'oversampled'], ckpt_info)

        if ckpt_info.generate_report:
            self.report_error_metrics(gts, pred_points, faces)


    def get_distortion_tensor(self, gt_points, pred_points, param, faces):
        distortion_gt   = self.compute_surface_distortion(gt_points, param, faces)
        distortion_pred = self.compute_surface_distortion(pred_points, param, faces)
        distortion      = torch.cat([distortion_gt, distortion_pred], dim=1)
        return distortion


    def distortion_to_dict(self, distortion):
        scalars_dict = {}
        ## copy distortion measures to dictionary
        scalars_dict['scale_gt'] = distortion[:, 0]
        scalars_dict['angle_gt'] = distortion[:, 1]
        scalars_dict['scale']    = distortion[:, 2]
        scalars_dict['angle']    = distortion[:, 3]
        return scalars_dict


    def report_surface_distortion(self, distortion, name):
        self.html_logs.add_entry('distortion_scale_gt', distortion[:,0], name)
        self.html_logs.add_entry('distortion_angle_gt', distortion[:,1], name)
        self.html_logs.add_entry('distortion_scale', distortion[:,2], name)
        self.html_logs.add_entry('distortion_angle', distortion[:,3], name)


    def report_error_metrics(self, gt_points, pred_points, faces):

        pt_distance = (gt_points - pred_points).pow(2).sum(-1)
        self.html_logs.add_entry('reconstruction', pt_distance, 'Full')

        try: # skip in case of import error
            dist1, dist2 = self.compute_chamfer_distance(gt_points, pred_points, faces)
            self.html_logs.add_entry('reconstruction', dist1.sqrt().squeeze(), 'Chamfer distance 1')
            self.html_logs.add_entry('reconstruction', dist2.sqrt().squeeze(), 'Chamfer distance 2')
            self.html_logs.add_entry('reconstruction', (dist1 + dist2).sqrt().squeeze(), 'Chamfer distance sum')
        except Exception as err:
            logging.error('ERROR computing chamfer distance: ' + str(err))
