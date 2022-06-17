
import torch
from torch.nn import functional as F
from torch.nn import Identity
from torch.nn import Module
from torch.nn import Sequential

from .utils import create_sequential_linear_layer
from .utils import get_init_fun


class MLP():
    ## MLP network for Neural Convolutional Surfaces

    def build_mlp(self, extend_w_uv=True):
        self.feat_size = self.cnn_struct['channels'][-1] # input number of features

        if extend_w_uv:
            self.feat_size += 2

        mlp_build_function = self.get_build_function(self.mlp_struct)

        self.mlp_out = mlp_build_function(self.mlp_struct, self.feat_size, self.output_size)

        init_fun = get_init_fun(self.mlp_struct['init'])
        self.mlp_out.apply(init_fun)

    def get_build_function(self, struct):

        mlp_build_function = self.build_sequential_mlp
        if struct.get('type', 'sequential') == 'residual':
            mlp_build_function = self.build_residual_mlp
        return mlp_build_function


    def build_sequential_mlp(self, struct, input_size, output_size):
        bias       = struct.get('bias', True) # use bias
        drop       = struct.get('drop', 0.0) # use bias
        norm       = struct.get('norm', None) # use bias
        act        = struct.get('act', 'ReLU')
        act_params = struct.get('act_params', {})

        layers = [ input_size ] + struct['layers'] + [ output_size ]

        return create_sequential_linear_layer(layers, act, norm, \
                            drop_prob=drop, last_act=False, bias=bias, act_params=act_params)

    def build_residual_mlp(self, struct, input_size, output_size):
        bias       = struct.get('bias', True) # use bias
        drop       = struct.get('drop', 0.0) # use bias
        norm       = struct.get('norm', None) # use bias
        act        = struct.get('act', 'ReLU')
        act_params = struct.get('act_params', {})

        layers = struct['layers']
        modules = []

        layer = create_sequential_linear_layer([input_size, layers[0]], act, norm, drop, bias, last_act=True, act_params=act_params)
        modules.extend([el for el in layer])

        ## create residual blocks
        for in_feats, out_feats in zip(layers[:-1], layers[1:]):
            block = ResidualMLPBlock(in_feats, act, norm, drop, bias, act_params, out_features=out_feats)
            modules.append(block)

        ## create last layer
        layer = create_sequential_linear_layer([layers[-1], output_size], act, norm, drop, bias, last_act=False)
        modules.extend([el for el in layer])

        ## assemble into a single sequential
        return Sequential(*modules)



class FineMLP(MLP):
    ## MLP network for Neural Convolutional surfaces
    ## add details on top of coarse shape

    def build_mlp(self, extend_w_uv=False):
        MLP.build_mlp(self, extend_w_uv=extend_w_uv)

        mlp_build_function = self.get_build_function(self.mlp_struct)
        self.mlp_coarse = mlp_build_function(self.mlp_coarse_struct, 2, self.output_size)

        init_fun = get_init_fun(self.mlp_coarse_struct['init'])
        self.mlp_coarse.apply(init_fun)


    def forward_mlp(self, uv_in, feats, return_uvs=False, only_uvs=False, return_displ=False):

        ## forward coarse mlp (coarse structure)
        coarse_pts = self.mlp_coarse(uv_in)

        if only_uvs:
            return coarse_pts

        ## forward fine mlp (details)
        displacement = self.mlp_out(feats)

        ## get normals for reference frame
        normals, jacobian = self.compute_normals(out=coarse_pts, wrt=uv_in, return_grad=True)

        ## compute reference frame using jacobian and normals
        a = F.normalize(jacobian[..., 0], dim=-1)
        b = torch.cross(normals, a, dim=-1)
        R = torch.stack([normals, a, b], dim=-1).reshape(-1, 3, 3) ## LRF (rotation matrix)

        ## rotate the displacement using LRF
        B = coarse_pts.size(0)
        N = coarse_pts.size(1)

        rot_disp = R.bmm(displacement.reshape(-1,3).unsqueeze(-1)).squeeze(-2)#.view(-1, 1, 3).bmm(R).squeeze(-2)
        if len(feats.size()) > 2:
            rot_disp = rot_disp.view(B, N, 3)

        ## add displacement on top of coarse structure
        out_pts = coarse_pts + rot_disp

        ## uncomment if you want to test displacement in normal direction only
        # out_pts = coarse_pts + normals*displacement

        if return_displ and return_uvs:
            return out_pts, coarse_pts, displacement
        if return_uvs:
            return out_pts, coarse_pts
        if return_displ:
            return out_pts, displacement
        return out_pts


class ResidualMLPBlock(Module):

    def __init__(self, in_features, act_fun, norm_layer, drop_prob, bias, act_params, out_features=None):
        super().__init__()

        layers = [in_features]*3
        if out_features is not None:
            layers[-1] = out_features

        layer = create_sequential_linear_layer(layers, act_fun, norm_layer, drop_prob, bias, last_act=True, act_params=act_params)

        self.shortcut = Identity()
        if in_features != out_features and out_features is not None:
            self.shortcut = create_sequential_linear_layer([in_features, out_features], act_fun, norm_layer, drop_prob, bias, last_act=False)

        self.residual = Sequential(*layer[:-1])
        self.post_act = layer[-1]


    def forward(self, x):
        x   = self.shortcut(x)
        res = self.residual(x)
        out = self.post_act(res + x)
        return out

