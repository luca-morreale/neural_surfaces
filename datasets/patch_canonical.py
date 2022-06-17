
import torch

from utils import sample_surface

from .mixin import DatasetMixin


class PatchCanonicalDataset(DatasetMixin):

    def __init__(self, config):

        self.sample_path   = config['sample_path']
        self.num_points    = config['num_points']

        self.sample = self.read_sample(self.sample_path)

        self.patches = self.sample['patches']
        self.shape   = self.sample['global']
        self.name    = self.shape['name']

        ## extract patch data
        self.extract_data_from_samples([self.sample])

        ## reposition patches (R and t)
        self.patch_centroids, self.patch_rotations = self.compute_canonical_transform(self.points, self.normals)
        self.patch_centroids = torch.stack(self.patch_centroids, dim=0)
        self.patch_rotations = torch.stack(self.patch_rotations, dim=0)

        ## get visual weights for interpolation
        if 'weights' not in self.sample['global']:
            self.assemble_correspondences([self.sample])
            self.compute_weights([self.sample])
            self.weights = self.weights[0]
        else:
            self.weights = self.sample['global']['weights']

        ## check if there a segmentation of the file
        if 'segmentation' in config:
            self.segmentation = self.read_sample(config['segmentation']).long()


    def __len__(self):
        return len(self.params)


    def __getitem__(self, index):
        ## sample patch surface
        params_to_sample = [self.params[index]]
        points, normals, params = sample_surface(self.num_points, self.points[index],
                                                self.faces[index], params_to_sample, method='pytorch3d')

        param = params[0]

        ## reposition patch surface (points)
        points  = (points - self.patch_centroids[index]).matmul(self.patch_rotations[index])
        normals = normals.matmul(self.patch_rotations[index])

        ## mask points if close to the boundary for normals estimation (assumes 2D param. is a square)
        mask = param.abs().max(dim=1)[0] < 0.99
        mask = mask.bool()

        data_dict = {
                    'param'   : param,
                    'gt'      : points,
                    'normals' : normals,
                    'mask'    : mask,
                    'Cs'      : self.Cs[index].view(1, 1),
                    'idx'     : torch.tensor([index]).long(),
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
        data_dict['Rs']           = self.patch_rotations
        data_dict['ts']           = self.patch_centroids
        data_dict['Cs']           = self.Cs
        data_dict['visual_W']     = self.weights
        data_dict['name']         = self.name
        data_dict['global_faces'] = self.shape['faces']
        data_dict['global_gt']    = self.shape['points']

        return data_dict
