
from .tensorboard import TensorboardLogger


def create(config, experiment):
    logger = globals()[config['name']](config)
    return logger