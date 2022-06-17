
import logging
import torch
from tqdm import trange

from utils import create_model_summary
from utils import summary_to_table


class TrainingLoop():

    def __init__(self, runner, num_epochs, **kwargs):
        self.runner = runner
        self.num_epochs = num_epochs
        self.__dict__.update(kwargs)


    def configure_run(self):
        ## get objects needed to run the experiment
        self.train_loader  = self.runner.train_loader()
        self.model         = self.runner.get_model()
        self.optimizers    = self.runner.get_optimizers()
        self.schedulers    = self.runner.get_schedulers()
        self.logger        = self.runner.get_logger()


    def run(self):

        self.configure_run()

        self.interrupted = False

        ## move model to device and update the runner
        self.model = self.runner.move_to_device(self.model)
        self.runner.model = self.model

        ## create model summary and print it (num weights and structure)
        model_summary = create_model_summary(self.model)
        summary = summary_to_table( { k:v for d in model_summary for k, v in d.items() } )
        print(summary)

        ## notify the training is starting
        with torch.no_grad():
            self.runner.train_starts()

        ## training loop
        try:
            self.loop()

            ## notify end training
            with torch.no_grad():
                self.runner.train_ends()

        except KeyboardInterrupt:
            logging.info('Detected KeyboardInterrupt, attempting graceful shutdown...')

            # user could press ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.interrupted = True
                with torch.no_grad():
                    self.runner.train_ends()


    def loop(self):

        ## training loop
        for epoch in trange(self.num_epochs):
            for batch in self.train_loader:

                self.zero_grads()

                batch = self.runner.move_to_device(batch) # move data to device
                loss, logs = self.runner.train_step(batch) # training iteration

                loss.backward()
                self.optimize() # optimize

                self.log_train(logs) # log data

            self.checkpointing(epoch) # checkpoint

            self.scheduling() # scheduling

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def optimize(self):
        for opt in self.optimizers:
            opt.step()

    def scheduling(self):
        for sch in self.schedulers:
            sch.step()

    def log_train(self, logs):
        self.logger.log_data(logs)

    def checkpointing(self, epoch):
        if (epoch+1) % self.checkpoint_epoch == 0:
            with torch.no_grad():
                self.runner.checkpoint(epoch)
