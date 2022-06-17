
import torch
from torch.nn import Module


class UnitCircleDomain(Module):

    ## Is there a way to avoid this? i.e. not calling this
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('one',  torch.tensor(1.0))


    def domain_mask(self, points2D, tris_points2D):

        mask_outside = points2D.pow(2).sum(-1) > 1.0
        mask_outside = mask_outside.bool()

        return ~mask_outside # mask inside


    def boundary_distances(self, points2D_target, points2D_source):
        mask = self.boundary_mask(points2D_source)

        boundary_distances = (points2D_target[mask].pow(2).sum(-1) - self.one)

        return boundary_distances

    def boundary_mask(self, points2D_source):

        with torch.no_grad():
            boundary = points2D_source.pow(2).sum(-1) > self.one - 1.0e-6
            mask = boundary.bool().squeeze()

        return mask

    def signed_distance_to_boundary(self, param):
        sd = param.pow(2).sum(-1) - 1.0
        return sd
