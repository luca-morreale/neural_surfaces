
from torch.nn import Module
from torch.nn import Sequential

from .modules.mlps import ResidualMLPBlock
from .modules.utils import create_sequential_linear_layer
from .modules.utils import get_init_fun


class MLP(Module):
    ## simple MLP network

    def __init__(self, config):
        super().__init__()

        in_size    = config['input_size'] # input size of MLP
        out_size   = config['output_size'] # output size of MLP
        layers     = config['layers'] # hidden layers size
        act_name   = config.get('act', 'Softplus') # activation for the MLP
        norm_name  = config.get('norm', None) # normalization layers (if any)
        drop_prob  = config.get('drop', 0.0) # dropout probability (if any)
        bias       = config.get('bias', True) # bias
        act_params = config.get('act_params', {}) # parameters for the activation function

        layers = [ in_size ] + layers + [ out_size ]

        self.mlp = create_sequential_linear_layer(layers, act_name, norm_name, drop_prob, bias, last_act=False, act_params=act_params)

        init_fun = get_init_fun(config['init'])
        self.mlp.apply(init_fun)


    def forward(self, x):
        return self.mlp(x)


class ResidualMLP(Module):

    def __init__(self, config):
        super().__init__()


        in_size    = config['input_size'] # input size of MLP
        out_size   = config['output_size'] # output size of MLP
        layers     = config['layers'] # hidden layers size
        act_name   = config.get('act', 'Softplus') # activation for the MLP
        norm_name  = config.get('norm', None) # normalization layers (if any)
        drop_prob  = config.get('drop', 0.0) # dropout probability (if any)
        bias       = config.get('bias', True) # bias
        act_params = config.get('act_params', {}) # parameters for the activation function

        modules = []

        ## create first layer
        layer = create_sequential_linear_layer([in_size, layers[0]], act_name, norm_name, drop_prob, bias, last_act=True, act_params=act_params)
        modules.extend([el for el in layer])

        ## create residual blocks
        for layer in layers:
            block = ResidualMLPBlock(layer, act_name, norm_name, drop_prob, bias, act_params)
            modules.append(block)

        ## create last layer
        layer = create_sequential_linear_layer([layers[-1], out_size], act_name, norm_name, drop_prob, bias, last_act=False)
        modules.extend([el for el in layer])

        ## assemble into a single sequential
        self.mlp = Sequential(*modules)

        ## initialize weights
        init_fun = get_init_fun(config['init'])
        self.mlp.apply(init_fun)


    def forward(self, x):
        x = self.mlp(x)
        return x

