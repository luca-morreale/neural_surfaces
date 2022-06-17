
import torch


def compute_distortion(points, faces, param):

    triangles = points[faces]
    triangles = triangles - triangles[:, -1].unsqueeze(1)
    triangles = triangles[:, :-1].transpose(1,2)

    U, _, _ = torch.svd(triangles, compute_uv=True)
    P = U.transpose(1,2).matmul(triangles)
    P_ext = torch.cat([P, torch.zeros(triangles.size(0), 2,1).to(P.device)], dim=-1)
    P_ext_2 = torch.cat([P_ext, torch.ones(P.size(0), 1, 3).to(P.device)], dim=1)
    P_inv = P_ext_2.inverse()

    Q = param[faces].transpose(1,2) # Fx2x3

    J = Q.matmul(P_inv)
    J = J[:, :2, :2]
    FFF = J.transpose(1,2).matmul(J)

    scale_distortion = FFF[:, 0, 0] / FFF[:, 1, 1]
    angle_distortion = FFF[:, 0, 0] * FFF[:, 1, 1]

    return scale_distortion, angle_distortion
