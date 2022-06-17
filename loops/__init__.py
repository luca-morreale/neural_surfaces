
from .modulating_loop import ModulatingTrainingLoop
from .training_loop import TrainingLoop
from .training_loop_gradient import GradientTrainingLoop

def create(name, kwargs):
    logger = globals()[name](**kwargs)
    return logger
