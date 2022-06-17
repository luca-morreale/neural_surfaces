
from torch.nn import Sequential

from .conv_block import ConvBlock
from .conv_block import ResidualConvBlock
from .utils import get_activation_function
from .utils import get_init_fun


class CNN():
    ## CNN block for Neural Convolutional Surfaces
    ## Upsample -> Conv2D -> Act (see ConvBlock)

    CONV_MODULE=ConvBlock

    def build_cnn(self):
        channels   = self.cnn_struct['channels'] # how many channels
        kernels    = self.cnn_struct['kernels'] # kernel size
        bias       = self.cnn_struct.get('bias', True) # use bias or not
        act_fun    = get_activation_function(self.cnn_struct['act'])
        act_params = self.cnn_struct.get('act_params', {}) # use bias or not

        in_channels = self.cnn_struct['latent_depth'] # latent code depth
        if ('use_positional', True) in self.structure.items():
            in_channels *= self.structure['num_positional']

        modules = []

        for i, out_ch in enumerate(channels):
            conv = self.CONV_MODULE(in_channels, out_ch, kernel_size=kernels[i], padding=int(kernels[i]/2),
                                    bias=bias, act_fun=act_fun, act_args=act_params)
            modules.append(conv)
            in_channels = out_ch
        self.cnn = Sequential(*modules)

        init_fun = get_init_fun(self.cnn_struct['init'])
        self.cnn.apply(init_fun)


class ResidualCNN(CNN):

    def build_cnn(self):

        self.CONV_MODULE = ResidualConvBlock
        super().build_cnn()
