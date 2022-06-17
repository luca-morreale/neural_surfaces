
from differential import DifferentialModule

from .base_cnn_map import CNNNSM
from .modules.convolutions import CNN, ResidualCNN
from .modules.interpolation import BatchInterpolation
from .modules.mlps import MLP, FineMLP


## All these classes are an extension from the base CNNNSM
## They glue together different CNNs and MLPs classes
## to form different classes w/o adding much code
## ie pieces are as modular as possible



###################
#     PATCHES     #
###################

class PCANeuralConvSurface(CNN, MLP, BatchInterpolation, CNNNSM):
    ## NCS with PCA and no coarse branch

    def build_mlp(self):
        return super().build_mlp(extend_w_uv=False)


    def forward_cnn(self, idx):
        code = self.embeddings(idx)
        return self._forward_cnn(code)

    def forward_mlp(self, x, feats):
        out3 = self.mlp_out(feats)
        return out3

    def forward(self, x, idx):
        out1 = self.forward_cnn(idx)
        out2 = self.forward_interpolation(x, out1)
        out3 = self.forward_mlp(x, out2)
        return out3



class NeuralConvSurface(DifferentialModule, FineMLP, PCANeuralConvSurface):
    ## NCS with coarse and fine branches

    def build_mlp(self):
        return FineMLP.build_mlp(self, extend_w_uv=False)

    def forward(self, uv, idx, global_uv, return_uvs=False, return_displ=False):
        out1 = self.forward_cnn(idx)
        out2 = self.forward_interpolation(uv, out1)
        out3 = self.forward_mlp(global_uv, out2, return_uvs=return_uvs, return_displ=return_displ)
        return out3

class NeuralResConvSurface(ResidualCNN, NeuralConvSurface):
    ## same as NeuralConvSurface but with Residual CNN
    pass
