
import os
import torch
import trimesh
from tqdm import trange

from checkpointing import AllCheckpointing
from checkpointing import ReportData
from utils import compute_distortion
from utils import faces_to_vertices

from .generic_runner import GenericRunner

class CheckpointRunner(AllCheckpointing, GenericRunner):

    def train_starts(self, model, experiment, checkpoint_dir):
        ## run a checkpoint iteration to save initialization and constant data
        ckpt_info = self.CKPTWrapper()
        ckpt_info.checkpoint_dir     = os.path.join(checkpoint_dir, 'init/')
        ckpt_info.save_constant_data = True
        self.run(model, experiment, ckpt_info)

    def run(self, model, experiment, ckpt_info):

        dataset = experiment['datasets']['train'].dataset

        for i in trange(dataset.num_checkpointing_samples()):

            self.reset_surface_storage()
            if ckpt_info.generate_report:
                self.create_report_data_obj()

            sample = dataset.get_checkpointing_sample(i)
            name   = sample['name']

            self.checkpoint_sample(sample, model, experiment, ckpt_info)

            self.end_checkpointing(model, name, ckpt_info)


    def checkpoint_sample(self, sample, model, experiment, ckpt_info):
        raise NotImplementedError()


    def reset_surface_storage(self):
        self.points3D_list = []
        self.faces_list    = []
        self.uvs_list      = []
        self.scalars_list  = []
        self.colors_list   = []
        self.names         = []


    def create_report_data_obj(self):
        self.html_logs = ReportData()
        self.html_logs.add_log('reconstruction', "Global reconstruction distance (L2)")
        self.html_logs.add_log('distortion_scale_gt', "GT scale distortion")
        self.html_logs.add_log('distortion_angle_gt', "GT angle distortion")
        self.html_logs.add_log('distortion_scale', "Scale distortion")
        self.html_logs.add_log('distortion_angle', "Angle distortion")


    def append_surface_data(self, points, faces, uvs, name_tags, ckpt_info, scalars=None, colors=None, constant=False):
        if (not constant) or (constant and ckpt_info.save_constant_data):
            self.points3D_list.append(points)
            self.faces_list.append(faces)
            self.uvs_list.append(uvs)
            self.scalars_list.append(scalars)
            self.colors_list.append(colors)
            constant = True
            name = self.build_prefix_name(name_tags, ckpt_info if not constant else None)
            self.names.append(name)

        self.timelapse_mesh(points, faces, uvs, name_tags, ckpt_info, constant=constant)


    def compute_surface_distortion(self, points, param, face):

        points = points.squeeze()
        scale, angle = compute_distortion(points, face, param.squeeze())
        angle = faces_to_vertices(points, face, angle, to_torch=True)
        scale = faces_to_vertices(points, face, scale, to_torch=True)

        distortion = torch.zeros([points.size(0), 2])
        distortion[:, 0] = scale
        distortion[:, 1] = angle

        return distortion


    def compute_chamfer_distance(self, gt_points, pred_points, faces):
        from chamfer.chamfer3D import chamfer_3DDist
        distChamfer = chamfer_3DDist()

        samples_gt,_ = trimesh.sample.sample_surface(trimesh.Trimesh(gt_points.cpu().numpy(), faces.cpu().numpy()), 100000)
        samples_our,_ = trimesh.sample.sample_surface(trimesh.Trimesh(pred_points.detach().cpu().numpy(), faces.cpu().numpy()), 100000)

        torch.set_grad_enabled(True)
        V = torch.from_numpy(samples_gt).float().unsqueeze(0)
        V_test = torch.from_numpy(samples_our).float().unsqueeze(0)
        V_test.requires_grad = True

        V = self.move_to_device(V)
        V_test = self.move_to_device(V_test)

        dist1, dist2, _, _ = distChamfer(V, V_test)
        torch.set_grad_enabled(False)

        return dist1, dist2


    def end_checkpointing(self, model, name, ckpt_info):
        self.save_surfaces_from_list(self.points3D_list, self.faces_list, self.uvs_list, self.scalars_list, self.colors_list, self.names, ckpt_info.checkpoint_dir)

        self.save_model(ckpt_info.checkpoint_dir, model, 0)

        if ckpt_info.generate_report:
            prefix_name = self.build_prefix_name([name], ckpt_info)
            self.save_report(self.html_logs.report_data, model, ckpt_info.training_time, ckpt_info.checkpoint_dir, prefix=prefix_name)
