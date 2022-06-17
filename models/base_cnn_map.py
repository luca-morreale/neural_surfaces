
import torch
from torch.nn import Module

from .modules import embeddings
from .modules.interpolation import Interpolation


class CNNNSM(Interpolation, Module):
    ## Base class for Neural Convolutional Surfaces
    ## allows to create variants of NCS

    def __init__(self, structure):
        super().__init__()
        self.init_interpolation()

        self.structure = structure

        self.input_size     = structure['input_size']
        self.output_size    = structure['output_size']

        self.latent_struct = structure['embeddings']
        self.cnn_struct    = structure['cnn'] # structure of cnn
        self.mlp_struct    = structure['mlp'] # structure of fine mlp

        if 'coarse_mlp' in structure:
            self.mlp_coarse_struct = structure['coarse_mlp'] ## structure for coarse mlp

        self.num_embeddings = 1
        if 'num_embeddings' in structure:
            self.num_embeddings = structure['num_embeddings'] ## info about embeddings

        self.build_cnn()
        self.build_mlp()
        self.build_latent()


    def build_cnn(self): # create CNN, child class will populate this
        raise NotImplementedError('build_cnn not implemented')

    def build_mlp(self): # create (all) MLPs, child class will populate this
        raise NotImplementedError('build_mlp not implemented')

    def build_latent(self): # create embeddings
        self.embeddings = embeddings.create(self.latent_struct)


    def forward(self, x): # base forward procedure
        out1 = self.forward_cnn()
        out2 = self.forward_interpolation(x, out1)
        out3 = self.forward_mlp(x, out2)
        return out3

    def forward_cnn(self):
        ## get codes and forward cnn
        code = self.embeddings()
        return self._forward_cnn(code)

    def _forward_cnn(self, code):
        ## forward cnn
        out = self.cnn(code)
        return out

    def forward_interpolation(self, x, feats):
        ## interpolate features
        return self.interpolate(feats, x)

    def forward_mlp(self, x, feats):
        ## forward fine mlp (mlp after cnn)
        out2 = torch.cat([feats, x], dim=-1)
        out3 = self.mlp_out(out2)
        return out3

    def positionalencoding(self, uvs):
        embed_fns = []
        for freq in self.pos_enc_w:
            embed_fns.append(torch.sin(uvs * freq))
            embed_fns.append(torch.cos(uvs * freq))
        return torch.cat(embed_fns, dim=-1)
