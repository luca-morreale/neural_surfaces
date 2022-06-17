
from torch.nn import Module

class Loss(Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)