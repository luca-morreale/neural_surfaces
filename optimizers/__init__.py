
from torch.optim import *


def create(config, experiment):
    optimizers = []
    for opt in config:
        ## optimizer for the embeddings
        if ('subset','embeddings') in opt.items():
            print('Optimizer -> embeddings')
            optimizers.append(globals()[opt['name']](experiment['models'].embeddings.parameters(), **opt['params']))

        ## optimizer for the coarse mlp (Convolutional Neural Surfaces)
        elif ('subset','coarse') in opt.items():
            print('Optimizer -> coarse')
            optimizers.append(globals()[opt['name']](experiment['models'].mlp_coarse.parameters(), **opt['params']))

        ## optimizer for the fine branch: embeddings + cnn + fine mlp (Convolutional Neural Surfaces)
        elif ('subset','fine') in opt.items():
            print('Optimizer -> fine')
            optimizers.append(globals()[opt['name']](list(experiment['models'].cnn.parameters()) + \
                                                    list(experiment['models'].mlp_out.parameters()) + \
                                                    list(experiment['models'].embeddings.parameters()), **opt['params']))
        ## optimizer for the entire model
        else:
            optimizers.append(globals()[opt['name']](experiment['models'].parameters(), **opt['params']))

    return optimizers