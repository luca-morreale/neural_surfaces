
from .training import *
from .checkpointing import *

def create(config, experiment):
    tasks = {}
    for k, v in config.items():
        if k == 'name' or k == 'params': continue

        if 'params' in config:
            tasks[k] = globals()[v](**config['params'])
        else:
            tasks[k] = globals()[v]()
    return tasks