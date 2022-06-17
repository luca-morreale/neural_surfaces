
import torch
from torch.nn import Module
from torch.nn import Softplus


class Sine(Module):

    def __init__(self, w0=1.0):
        super().__init__()
        self.register_buffer('w0', torch.FloatTensor([w0]))

    def forward(self, x):
        return torch.sin(self.w0*x)


class Cosine(Module):

    def __init__(self, w0=1.0):
        super().__init__()
        self.register_buffer('w0', torch.FloatTensor([w0]))

    def forward(self, x):
        return torch.cos(self.w0*x)


class SquarePlus(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x + (x.pow(2) + 4).sqrt()) / 2.0


class SoftPlusZero(Softplus):
    def __init__(self):
        super().__init__()
        self.register_buffer('translation', torch.log(torch.tensor([2.0])))

    def forward(self, x):
        return super().forward(x) - self.translation

