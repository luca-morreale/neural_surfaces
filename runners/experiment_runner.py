
import json

from loops import create as create_loop

from .generic_runner import GenericRunner
from .path_generator import ConfigPathGenerator


class ExperimentRunner(GenericRunner, ConfigPathGenerator):

    def __init__(self, config, modules_creator):
        super().__init__()

        with open(config) as json_file:
            config_text = json.load(json_file)
        self.config = config_text
        self.modules_creator = modules_creator

        self.compose_config_path()
        # create experiment objects (eg dataset, model, ...)
        self.experiment = self.modules_creator.create_experiment_modules(self.config)


    def run_loop(self):
        kwargs = self.config['loop']
        kwargs['runner'] = self
        self.loop = create_loop(self.config['loop']['name'], kwargs) # create the training loop object

        self.experiment['loss'] = self.move_to_device(self.experiment['loss'])

        self.loop.run()


    def train_loader(self):
        return self.experiment['datasets']['train']

    def get_model(self):
        return self.experiment['models']

    def get_optimizers(self):
        return self.experiment['optimizers']

    def get_schedulers(self):
        return self.experiment['schedulers']

    def get_logger(self):
        return self.experiment['logging']

    def train_step(self, batch):
        return self.experiment['tasks']['train'].run(batch, self.model, self.experiment)

    def val_step(self, batch):
        return self.experiment['tasks']['val'].run(batch, self.model, self.experiment)

    def test_step(self, batch):
        return self.experiment['tasks']['test'].run(batch, self.model, self.experiment)

    def checkpoint(self, epoch):
        return self.experiment['tasks']['checkpoint'].run(self.model, self.experiment, epoch)
