
import torch
from torch.nn import Module


class TrisDomain(Module):

    def __init__(self, **kwargs):
        super().__init__()

    def domain_mask(self, points2D_target, tris_points2D):
        '''
        points2D_target: Nx2   points 2D in the target domain
        tris_points2D:   Mx3x2 points 2D aggregated based on faces definint the domain

        create binary mask for which points in 'points2D_target' are contained in the 2D domain
        defined by 'tris_points2D'
        '''
        v1 = tris_points2D[:, 0]
        v2 = tris_points2D[:, 1]
        v3 = tris_points2D[:, 2]

        point_mask = self.points_in_triangle(points2D_target.view(-1, 1, 2),
                                v1[None, :], v2[None, :], v3[None, :])

        point_mask = point_mask.sum(dim=-1).bool() # point is contained in at least a triangle

        return point_mask


    def points_in_triangle(self, pts, v1, v2, v3):
        ### check pts is contained in the triangles

        d1 = self.sign(pts, v1, v2)
        d2 = self.sign(pts, v2, v3)
        d3 = self.sign(pts, v3, v1)

        has_neg = (d1 < 0) + (d2 < 0) + (d3 < 0)
        has_pos = (d1 > 0) + (d2 > 0) + (d3 > 0)

        return ~(has_neg * has_pos)


    def sign(self, p1, p2, p3):
        return (p1[..., 0] - p3[...,  0]) * (p2[..., 1] - p3[..., 1]) \
                - (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])

    def compute_tris_area(self, tris):
        A = tris[:, 0]
        B = tris[:, 1]
        C = tris[:, 2]

        AB = B - A
        AC = C - A
        area = torch.cross(AB, AC).pow(2).sum(-1) / 2.0
        return area

    def signed_distance_to_boundary(self, param):
        sd = param.pow(2).sum(-1) - 1.0
        return sd
