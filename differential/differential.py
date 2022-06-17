
import torch
from torch import autograd as Grad
from torch.nn import functional as F
from torch.nn import Module

from torch_batch_svd import svd


class DifferentialModule(Module):

    # ================================================================== #
    # =================== Compute the gradient ========================= #
    def gradient(self, out, wrt):

        B = 1 if len(out.size()) < 3 else out.size(0)
        N = out.size(0) if len(out.size()) < 3 else out.size(1)
        R = out.size(-1)
        C = wrt.size(-1)

        gradients = []
        for dim in range(R):
            out_p = out[..., dim].flatten()

            select = torch.ones(out_p.size(), dtype=torch.float32).to(out.device)

            gradient = Grad.grad(outputs=out_p, inputs=wrt, grad_outputs=select, create_graph=True)[0]
            gradients.append(gradient)

        if len(out.size()) < 3:
            J_f_uv = torch.cat(gradients, dim=1).view(N, R, C)
        else:
            J_f_uv = torch.cat(gradients, dim=2).view(B, N, R, C)

        return J_f_uv

    def backprop(self, out, wrt):

        select = torch.ones(out.size(), dtype=torch.float32).to(out.device)

        J = Grad.grad(outputs=out, inputs=wrt, grad_outputs=select, create_graph=True)[0]
        J = J.view(wrt.size())
        return J

    # ================================================================== #
    # ================ Compute normals using gradient ================== #
    def compute_normals(self, jacobian=None, out=None, wrt=None, return_grad=False):

        if jacobian is None:
            jacobian = self.gradient(out=out, wrt=wrt)

        cross_prod = torch.cross(jacobian[..., 0], jacobian[..., 1], dim=-1)

        # set small normals to zero, happens only when vectors are orthogonal
        idx_small = cross_prod.pow(2).sum(-1) < 10.0**-7
        # normals
        normals = F.normalize(cross_prod, p=2, dim=-1)  # (..., N, 3)
        normals[idx_small] = cross_prod[idx_small]

        if return_grad:
            return normals, jacobian
        return normals

    # ================================================================== #
    # ================ Compute first fundamental form ================== #
    def compute_FFF(self, jacobian=None, out=None, wrt=None, return_grad=False):
        if jacobian is None:
            jacobian = self.gradient(out=out, wrt=wrt)

        B = jacobian.size(0)
        N = jacobian.size(1)

        # 1st fundamental form (g)
        g = torch.matmul(jacobian.transpose(-2, -1), jacobian)
        if len(jacobian.size()) <= 3: # num_pts x 3 x 2 (no batching)
            g = g.reshape(B, 4)
        else:
            g = g.reshape(B, N, 4)
        # First fundamental form shaped as BxNx4 where last dimension is [E, F, F, G].

        # Extracts E, F, G terms to not keep all matrix
        E = g[..., 0]
        F = g[..., 1]
        G = g[..., 3]

        if return_grad:
            return E, F, G, jacobian
        return E, F, G

    # ================================================================== #
    # ================ Compute second fundamental form ================= #
    def compute_SFF(self, jacobian=None, out=None, wrt=None, return_grad=False, return_normals=False):
        normals, jacobian = self.compute_normals(jacobian=jacobian, out=out, wrt=wrt, return_grad=True)

        ru = jacobian[..., 0]
        rv = jacobian[..., 1]

        Jru = self.gradient(out=ru, wrt=wrt)
        Jrv = self.gradient(out=rv, wrt=wrt)
        ruu = Jru[..., 0]
        ruv = Jru[..., 1]
        rvv = Jrv[..., 1]

        L = (ruu * normals).sum(-1)
        M = (ruv * normals).sum(-1)
        N = (rvv * normals).sum(-1)

        if return_grad and return_normals:
            return L, M, N, jacobian, normals
        if return_grad:
            return L, M, N, jacobian
        return L, M, N

    # ================================================================== #
    # ================ Compute curvature analytically ================== #
    def compute_curvature(self, jacobian=None, out=None, wrt=None):
        L, M, N, jacobian = self.compute_SFF(jacobian=jacobian, out=out, wrt=wrt, return_grad=True, return_normals=False)
        E, F, G = self.compute_FFF(jacobian=jacobian)

        mean_curv  = 0.5 * (G * L - 2.0 * F * M + E * N) / (E * G - F.pow(2))
        gauss_curv = (L * N - M.pow(2)) / (E * G - F.pow(2))

        return mean_curv, gauss_curv

    # install https://github.com/KinglittleQ/torch-batch-svd
    # ================================================================== #
    # ================= Invert Rectangular Jacobians =================== #
    def invert_J(self, J):
        J_ = self.reduce_J(J)
        J_inv = J_.inverse()

        return J_inv

    def left_invese_J(self, J):
        U, e, V = svd(J)
        J_inv = V.matmul(torch.diag_embed(1.0/e)).matmul(U.transpose(1,2))

        return J_inv

    def reduce_J(self, J):
        # reduce J to a 2x2 using the SVD

        U, e, V = svd(J)
        J_ = J.transpose(1,2).matmul(U).transpose(1,2)

        return J_
