
import numpy as np
import torch

from .distortion import DistortionMixin


class SurfaceMapLoss(DistortionMixin):
    reg_distortion      = 0.0
    reg_folding         = 0.0
    reg_outside         = 0.0
    reg_outside_folding = 0.0
    sharp_value      = 1.0
    smoothing_epochs = 1.0
    smoothing_value  = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.smoothing_step = np.exp(1 / self.smoothing_epochs * np.log(self.sharp_value/self.smoothing_value))

    def compute_per_point_distortion(self, target_points3D, target_points2D, source_points2D, source_points3D):
        FFF, J_h = self.compute_differential_quantities(target_points3D, target_points2D, source_points2D, source_points3D)
        point_distortion = self.map_distortion(FFF)

        return point_distortion, J_h

    def forward(self, target_points3D, target_points2D, source_points2D, source_points3D, domain_mask=None):

        point_distortion, J_h = self.compute_per_point_distortion(target_points3D, target_points2D, source_points2D, source_points3D)
        fold_penalty          = self.fold_regularization(J_h)

        if domain_mask is not None:
            point_distortion     = point_distortion[domain_mask]
            fold_penalty_inside  = fold_penalty[domain_mask]


        ############## Aggregation ##############
        loss_distortion      = point_distortion.mean() # avg based on points falling inside domain
        loss_folding_inside  = fold_penalty_inside.mean()


        ############## Stats ##############
        median_distortion = point_distortion.median()
        geo_grad = self.backprop(loss_distortion, J_h)
        if domain_mask is not None:
            geo_grad = geo_grad[domain_mask]


        ############## Loss ##############
        loss = self.reg_distortion * loss_distortion + \
                self.reg_folding * loss_folding_inside

        logs = {'loss':                 loss.detach(),
                'loss_distortion':      loss_distortion.detach(),
                'loss_folding':         loss_folding_inside.detach(),
                'median_distortion':    median_distortion.detach(),
                'geometric_grad_med':   geo_grad.abs().median(),
                'geometric_grad_avg':   geo_grad.abs().mean()
        }

        return loss, logs
