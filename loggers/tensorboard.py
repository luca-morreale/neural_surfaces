
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger():
    ## log data on tensorboard

    def __init__(self, config):

        self.logger = SummaryWriter(config['folder'])
        self.namespace = config['namespace'] # this is a prefix for the experiment
        self.epochs = {}

    def log_data(self, data_dict):

        for k, v in data_dict.items():
            it = self.epochs.get(k, 0)
            self.epochs[k] = it + 1

            self.logger.add_scalar('{}/{}'.format(self.namespace, k), v, it)

    def log_histograms(self, data_dict):
        ## quite slow and data consuming, better avoid
        for k, v in data_dict.items():
            it = self.epochs.get(k, 0)
            self.epochs[k] = it + 1

            self.logger.add_histogram('{}/{}'.format(self.namespace, k), v, it)
