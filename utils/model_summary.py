
import numpy as np
from prettytable import PrettyTable

'''
summary = create_model_summary(model)
table   = summary_to_table( {k: v for d in summary[:2] for k, v in d.items()} )
print(table)
'''

def create_model_summary(model):

    summary = dict((name, LayerInfo(module)) for name, module in model.named_modules())

    max_depth = 0
    for k in summary.keys():
        depth = k.count(".")
        if depth > max_depth:
            max_depth = depth

    summary_by_level = [ {} for _ in range(max_depth+1) ]
    for k in summary.keys():
        depth = k.count(".")
        summary_by_level[depth][k] = summary[k]

    return summary_by_level


def summary_to_table(summary):
    table = PrettyTable(['Name', 'Type', 'Params'])
    table.align["Name"] = "l"
    table.align["Type"] = "l"
    table.align["Params"] = "r"
    for k in sorted(summary.keys()):
        table.add_row([k, summary[k].layer_type, summary[k].num_parameters])

    return table

class LayerInfo(object):

    def __init__(self, module):
        self.module = module

    @property
    def layer_type(self):
        return str(self.module.__class__.__name__)

    @property
    def num_parameters(self):
        return sum(np.prod(p.shape) for p in self.module.parameters())
