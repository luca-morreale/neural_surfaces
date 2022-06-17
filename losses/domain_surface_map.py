
from .domain import *
from .mixin import Loss
from .surface_map import *


class DomainSurfaceMapLoss(Loss):

    reg_boundary = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        surf_map_name = kwargs['surf_map']
        domain_name = kwargs['domain']

        self.surf_map_loss = globals()[surf_map_name](**kwargs)
        self.domain = globals()[domain_name](**kwargs)


    def forward(self, target_points3D, target_points2D, source_points2D, source_points3D, target_domain):

        domain_mask = self.domain.domain_mask(target_points2D, target_domain)
        loss_distortion, logs = self.surf_map_loss(target_points3D, target_points2D, source_points2D, source_points3D, domain_mask)

        points_outside = target_points2D[~domain_mask]
        loss_domain = points_outside.pow(2).sum(-1).mean() if points_outside.nelement() != 0 else 0.0

        loss = loss_distortion + self.reg_domain * loss_domain

        logs['loss_domain'] = loss_domain.detach() if points_outside.nelement() != 0 else loss_domain
        logs['loss']        = loss.detach()

        return loss, logs


    def boundary_loss(self, source_boundary_target, source_boundary):

        boundary_distances = self.domain.boundary_distances(source_boundary_target, source_boundary)
        loss_boundary = boundary_distances.pow(2).mean()

        logs = { 'loss': loss_boundary }

        return loss_boundary, logs
