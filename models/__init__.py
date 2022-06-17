
import torchvision.models as models

from .cnn_map import NeuralConvSurface
from .cnn_map import NeuralResConvSurface
from .cnn_map import PCANeuralConvSurface
from .mlp import MLP
from .mlp import ResidualMLP
from .neural_map import NeuralMap
from .neural_map import ParametrizationMap

def create(config, experiment):
    model = globals()[config['name']](config['structure'])

    return model
