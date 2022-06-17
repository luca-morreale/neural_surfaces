
from .domain_surface_map import DomainSurfaceMapLoss
from .mse import MSELoss, MAELoss
from .ssd import SSDLoss

def create(config, experiment):
    loss = globals()[config['name']](**config['params'])

    return loss
