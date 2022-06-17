
import torch

from utils import sample_surface

from .pca_patch import PCAPatchDataset


class PatchGlobalDataset(PCAPatchDataset):

    def __init__(self, config):
        super().__init__(config)

        self.coarse_param = self.shape['param']


    def __getitem__(self, index):

        coarse_param = self.coarse_param[self.param_idxs[index]]

        ## sample patch surface (and coarse parametrization)
        params_to_sample = [self.params[index], coarse_param]
        points, normals, params = sample_surface(self.num_points, self.points[index],
                                                self.faces[index], params_to_sample, method='pytorch3d')#,
                                                #weights=self.F_weights[index])

        param = params[0]
        coarse_param = params[1]

        ## mask points if close to the boundary for normals estimation (assumes 2D param. is a square)
        mask = param.abs().max(dim=1)[0] < 0.99 # not too close to the border of the patch
        mask = mask.bool()

        ### uncomment IF you want to sample only on vertices
        # point_batches = torch.randperm(self.params[index].size(0))[:self.num_points]

        # n = self.num_points - point_batches.size(0)
        # if n > 0:
        #     extra_sample = (torch.ones(self.params[index].size(0))).multinomial(num_samples=n, replacement=True)
        #     point_batches = torch.cat([point_batches, extra_sample]).reshape(-1).long()

        # param = self.params[index][point_batches]
        # points = self.points[index][point_batches]
        # coarse_param = self.shape['param'][self.param_idxs[index][point_batches]]


        data_dict = {
                    'param'        : param,
                    'gt'           : points,
                    'normals'      : normals,
                    'mask'         : mask,
                    'coarse_param' : coarse_param,
                    'idx'          : torch.tensor([index]).long(),
                }

        return data_dict

    def num_checkpointing_samples(self):
        return 1

    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['num_patches']  = len(self.params)
        data_dict['param']        = self.params
        data_dict['gts']          = self.points
        data_dict['faces']        = self.faces
        data_dict['param_idxs']   = self.param_idxs
        data_dict['idxs']         = torch.arange(len(self.params))
        data_dict['visual_W']     = self.weights
        data_dict['name']         = self.name
        data_dict['global_faces'] = self.shape['faces']
        data_dict['global_gt']    = self.shape['points']
        data_dict['coarse_param'] = self.coarse_param

        return data_dict
