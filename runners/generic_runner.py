
import torch
from torch import tensor
from torch import Tensor
from torch.nn import Module


class GenericRunner(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        ## set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ## check pytorch geometry is installed (not required, check to move things on cuda)
        try:
            from torch_geometric.data import Batch
            self.pyg_batch = Batch
            self.has_pyg = True
        except ImportError:
            self.has_pyg = False

        ## create wrapper class to store info for checkpointing
        class Wrapper:
            generate_report    = False
            training_time      = 0.0
            checkpoint_dir     = ''
            epoch              = -1
            save_constant_data = False
        self.CKPTWrapper = Wrapper


    def get_device(self):
        return self.device


    def move_to_device(self, data):
        if isinstance(data, Module) or type(data) == tensor or type(data) == Tensor:
            data = data.to(self.device)
        elif type(data) == dict:
            for k, v in data.items():
                data[k] = self.move_to_device(v) # recursion for subelements
        elif type(data) == list or type(data) == tuple:
            data = [ self.move_to_device(el) for el in data ] # recursion for subelements
        elif self.has_pyg:
            if type(data) == self.pyg_batch:
                data = data.to(self.device)
        return data
