
import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter


def create(config):
    embeddings = globals()[config['name']](**config['params'])
    return embeddings


class Embedding(Module):
    ## latent codes

    def __init__(self, num_embeddings, C, H, W):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.C = C
        self.H = H
        self.W = W
        code = torch.randn([num_embeddings, C, H, W])*0.0
        ## feel free to try different initializations
        # code = torch.zeros([num_embeddings, C, H, W])

        self.register_parameter('code', Parameter(code.float()))

    def forward(self, idx=None):

        if idx is None: # if no idx is given then return all codes
            return self.code

        selected = self.code[idx]

        return selected.view(-1, self.C, self.H, self.W)
