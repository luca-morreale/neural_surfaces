
import torch

from differential import DifferentialModule


class DistortionMixin(DifferentialModule):

    ## Is there a way to avoid this? i.e. not calling this
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('eye', torch.eye(2))
        self.register_buffer('eps',  torch.tensor(1.0e-8))
        self.register_buffer('one',  torch.tensor(1.0))


    def conformal(self, FFF):
        E = FFF[:, 0,0]
        G = FFF[:, 1,1]

        ### conformal: || _lambda * M - I ||
        lambd = (E + G) / FFF.pow(2).sum(-1).sum(-1)
        ppd   = (lambd.view(-1, 1, 1) * FFF - self.eye).pow(2).sum(-1).sum(-1)
        return ppd


    def symmetric_dirichlet(self, FFF):
        FFtarget_inv = (FFF + self.eps*self.eye).inverse()

        E = FFF[:, 0,0] - 1.0
        G = FFF[:, 1,1] - 1.0
        E_inv = FFtarget_inv[:, 0,0] - 1.0
        G_inv = FFtarget_inv[:, 1,1] - 1.0

        ### symm dirichlet: trace(J^T J) + trace(J_inv^T J_inv )
        dirichlet = E + G
        inv_dirichlet = E_inv + G_inv
        ppd = dirichlet + inv_dirichlet
        return ppd


    def arap(self, FFF):
        ppd = (FFF - self.eye).pow(2).sum(-1).sum(-1)
        return ppd


    def equiareal(self, FFF):
        E = FFF[:, 0,0]
        G = FFF[:, 1,1]

        ### equi-areal: E*G = 1
        ppd = (E * G) - self.one
        ppd = ppd.pow(2) + (1.0 / (ppd + self.eps)).pow(2)

        return ppd


    def fold_regularization(self, J):
        J_det = J.det()
        J_det_sign = torch.sign(J_det)
        pp_fold = torch.max(-J_det_sign * torch.exp(-J_det), self.zero)

        return pp_fold


    def compute_differential_quantities(self, target_points3D, target_points2D, source_points2D, source_points3D):
        '''
            target_points3D : Nx3 3D points on target surface
            target_points2D : Nx2 2D points in target surface domain
            source_points2D : Nx2 2D points in source surface domain
            source_points3D : Nx3 3D points on source surface
        '''

        J_f  = self.gradient(out=target_points3D, wrt=target_points2D)
        J_h  = self.gradient(out=target_points2D, wrt=source_points2D)
        J_fh = J_f.matmul(J_h)

        J_g     = self.gradient(out=source_points3D, wrt=source_points2D)
        J_g_inv = self.invert_J(J_g)

        J = J_fh.matmul(J_g_inv)

        # First Fundamental Form
        FFF = J.transpose(1,2).matmul(J)

        return FFF, J_h


    def compute_geometric_gradient(self, loss, J_h):
        geo_grad = self.backprop(loss, J_h)
        geo_grad = geo_grad.pow(2).sum(-1).sum(-1).mean()

        return geo_grad
