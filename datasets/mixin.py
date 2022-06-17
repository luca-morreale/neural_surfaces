
import numpy as np
import pickle
import torch


class DatasetMixin:

    def __len__(self):
        return self.num_epochs if not self.validation else 1

    def read_sample(self, path):
        if path[-3:] == 'pkl':
            sample = self.read_pickle_sample(path)
        elif path[-3:] == 'pth':
            sample = self.read_torch_sample(path)
        else:
            sample = self.read_numpy_sample(path)
        return sample

    def read_pickle_sample(self, path):
        return pickle.load(open(path, 'rb'))

    def read_torch_sample(self, path):
        return torch.load(path, map_location='cpu')

    def read_numpy_sample(self, path):
        return torch.from_numpy(np.load(path))

    # read a saved surface map (.pth file)
    def read_map_sample(self, path):
        sample = self.read_torch_sample(path)

        sample['faces'] = sample['faces'].long()
        sample['visual_faces'] = sample['visual_faces'].long()
        sample['C'] = None if 'C' not in sample else sample['C']

        for k in sample['weights'].keys():
            sample['weights'][k].requires_grad = False
            sample['weights'][k] = sample['weights'][k].cpu()

        return sample

    # split a set of points into 'num_blocks' batches
    def split_to_blocks(self, size, num_blocks):
        idxs = torch.randperm(size)
        block_size = int(float(idxs.size(0)) / float(num_blocks))
        blocks = []
        for i in range(num_blocks):
            blocks.append(idxs[block_size * i : block_size * (i + 1)])

        return blocks

    def compute_lands_rotation(self, lands_source, lands_target):
        ### compute rotation matrix

        with torch.no_grad(): # not sure if this is necessary
            # R * X^T = Y
            center_lands_source = lands_source - lands_source.mean(dim=0)
            center_lands_target = lands_target - lands_target.mean(dim=0)
            H = center_lands_source.transpose(0,1).matmul(center_lands_target)
            u, e, v = torch.svd(H)
            R = v.matmul(u.transpose(0,1)).detach()

            # check rotation is not a reflection
            if R.det() < 0.0:
                v[:, -1] *= -1
                R = v.matmul(u.transpose(0,1)).detach()

        t = lands_target.mean(dim=0) - lands_source.mean(dim=0).matmul(R.t())

        return R, t


    def compute_canonical_transform(self, points, normals):
        ### PCA transformation to rotate and tranlate patches to the origin
        patch_centroids = [ patch.mean(dim=0) for patch in points ]

        patch_rotations = []
        for i, patch in enumerate(points):
            R = self.rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), normals[i].mean(dim=0))

            patch_rotations.append(torch.from_numpy(R).float())

        return patch_centroids, patch_rotations

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix


    def assemble_correspondences(self, samples):
        ### create correspondance from patches to global shape
        self.corresp = []
        for shape in samples:
            ptx  = [ [] for _ in shape['global']['points'] ]
            for i, patch in enumerate(shape['patches']):
                for j, idx in enumerate(patch['idx']):
                    ptx[idx].append([i, j])
            self.corresp.append(ptx)


    def compute_weights(self, samples):
        ### compute visualization weights based on parametrization location
        self.weights = []
        for i, shape in enumerate(samples):

            weights = [ 1.0-item['param'].abs().max(dim=1)[0] for item in shape['patches'] ]

            for el in self.corresp[i]:
                if len(el) <= 1:
                    weights[el[0][0]][el[0][1]] = 1.0
                w_sum = 0
                for i, j in el:
                    w_sum += weights[i][j]
                if w_sum < 1.0e-6:
                    for i, j in el:
                        weights[i][j] = 1.0 / len(el)
            self.weights.append(weights)

    def extract_data_from_samples(self, samples):
        ### read data of patches and put them into lists
        self.points     = self.get_data_from_samples_list(samples, 'points')
        self.params     = self.get_data_from_samples_list(samples, 'param')
        self.faces      = self.get_data_from_samples_list(samples, 'faces')
        self.normals    = self.get_data_from_samples_list(samples, 'normals')
        self.Cs         = torch.tensor(self.get_data_from_samples_list(samples, 'C')).float()
        self.param_idxs = self.get_data_from_samples_list(samples, 'idx')

    def get_data_from_samples_list(self, samples, key):
        return [ item[key] for shape in samples for item in shape['patches'] ]
