
import torch

def tensor_to_numpy(mat, squeeze=False):
    if type(mat) == torch.Tensor or type(mat) == torch.nn.Parameter:
        mat = mat.detach().cpu().numpy()
        if squeeze:
            mat = mat.squeeze()

    return mat
