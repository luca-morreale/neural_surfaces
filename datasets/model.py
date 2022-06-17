
import torch
from math import ceil

from utils import sample_surface

from .mixin import DatasetMixin


class ModelDataset(DatasetMixin):

    def __init__(self, config):

        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)

        ## read sample split into patches (look only at global)
        if 'global' in self.sample.keys():
            self.points = self.sample['global']['points'].float()
            self.param  = self.sample['global']['param'].float()
            self.faces  = self.sample['global']['faces'].long()
            self.name   = self.sample['global']['name']
            self.normals = torch.zeros_like(self.points)
            self.visual_param = None
            self.visual_faces = None

        ## read surface map sample
        else:
            self.points        = self.sample['points'].float()
            self.param         = self.sample['param'].float()
            self.faces         = self.sample['faces'].long()
            self.normals       = self.sample['normals'].float()
            self.visual_param  = self.sample['oversampled_param'].float()
            self.visual_faces  = self.sample['oversampled_faces'].long()
            self.name          = self.sample['name']

        ## split into batches
        self.num_batches = ceil(self.points.size(0) / self.num_points)
        self.batchs_idx = self.split_to_blocks(self.points.size(0), self.num_batches)

        ## check how mask normals
        self.mask_normals = True if 'mask_normals' not in config else config['mask_normals']
        if self.mask_normals:
            self.mask_type = config['mask_normals_type']


    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):

        ## get current batch
        idx     = self.batchs_idx[index%self.num_batches]
        points  = self.points[idx]
        params  = self.param[idx]
        normals = self.normals[idx]
        N = normals.size(0)

        ## sample extra points from the surface
        params_to_sample = [self.param]
        P, n, p = sample_surface(self.num_points, self.points,
                                                self.faces, params_to_sample, method='pytorch3d')

        ## concat sampled points
        params  = torch.cat([params, p[0]], dim=0)
        points  = torch.cat([points, P], dim=0)
        normals = torch.cat([normals, n], dim=0)

        ## mask normals
        mask = torch.ones(params.size(0)).bool()
        if self.mask_normals:
            if self.mask_type == 'circle':
                mask = ~(params.pow(2).sum(-1) < 0.99).bool() # mask for normals
            elif self.mask_type == 'square':
                mask = ~(params < 0.99).prod(-1).bool() # mask for normals
        mask[:N] = False # do not enforce normal loss to vertices

        data_dict = {
                'param':   params,
                'gt':      points,
                'normals': normals,
                'mask':    mask
        }

        return data_dict


    def num_checkpointing_samples(self):
        return 1


    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['param'] = self.param
        data_dict['gts']   = self.points
        data_dict['faces'] = self.faces
        data_dict['name']  = self.name

        if self.visual_param is not None:
            data_dict['oversampled_param'] = self.visual_param
            data_dict['oversampled_faces'] = self.visual_faces

        return data_dict
