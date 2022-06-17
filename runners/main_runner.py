
import datetime
import logging

from checkpointing import BlenderCheckpointing

from .experiment_runner import ExperimentRunner


class MainRunner(ExperimentRunner, BlenderCheckpointing):
    ## more advanced version of the Generic runner

    def __init__(self, config, modules_creator):
        super().__init__(config, modules_creator)
        self.checkpoint_dir = self.config['checkpointing']['folder']


    def train_starts(self):

        ## check what kind of libraries are there and remove checkpointing that cannot be used
        self.check_all_imports()

        ## if the train runner need to do something at startup call it
        train_fun = getattr(self.experiment['tasks']['train'], "train_starts", None)
        if callable(train_fun):
            self.experiment['tasks']['train'].train_starts(self.model, self.experiment, self.checkpoint_dir)

        ## same for checkpoint runner
        checkp_fun = getattr(self.experiment['tasks']['checkpoint'], "train_starts", None)
        if callable(checkp_fun):
            self.experiment['tasks']['checkpoint'].train_starts(self.model, self.experiment, self.checkpoint_dir)

        ## record time of start
        logging.info('Training starts')
        self.start_time = datetime.datetime.now()


    def train_ends(self):
        ## record time of end
        self.end_time = datetime.datetime.now()
        self.training_time = self.end_time - self.start_time
        logging.info('Training ends')

        ## checkpoint experiment with full report
        ckpt_info = self.CKPTWrapper()
        ckpt_info.generate_report = True
        ckpt_info.training_time   = self.training_time
        ckpt_info.checkpoint_dir  = self.checkpoint_dir
        ckpt_info.epoch           = int(1.0e10)
        self.experiment['tasks']['checkpoint'].run(self.model, self.experiment, ckpt_info)

        ## render everything if possible
        self.render()


    def checkpoint(self, epoch):
        ## normal checkpointing
        ckpt_info = self.CKPTWrapper()
        ckpt_info.checkpoint_dir = self.checkpoint_dir
        ckpt_info.epoch          = epoch
        return self.experiment['tasks']['checkpoint'].run(self.model, self.experiment, ckpt_info)

    def check_all_imports(self):
        import inspect
        for base_class in inspect.getmro(MainRunner):
            if 'check_imports' in base_class.__dict__.keys():
                base_class.check_imports(self)
