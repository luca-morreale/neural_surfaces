
import logging
from statistics import median
from tqdm import trange

from .training_loop import TrainingLoop


class GradientTrainingLoop(TrainingLoop):
    ### used in Neural Surface Maps


    def compute_gradient(self):
        ## compute the median gradient for all parameters (median of mean, easy aggregation and does not clutter logging)
        gradient = []
        for _, params in self.model.named_parameters():
            if params.grad is not None:
                tmp_grad = params.grad.abs()
                gradient.append(tmp_grad.mean())

        return median(gradient)


    def loop(self):

        num_samples = len(self.train_loader)

        ## training loop
        for epoch in trange(self.num_epochs):

            grad = 0
            for batch in self.train_loader:

                converged = False

                self.zero_grads()

                batch = self.runner.move_to_device(batch)
                loss, logs = self.runner.train_step(batch)

                loss.backward()

                gradient = self.compute_gradient()
                grad += gradient

                self.optimize()

                self.log_train(logs)

                ## check if the gradient has vanished or below
                ## threshold then model has converged and can stop
                if gradient < self.grad_stop:
                    converged = True
                    break

            ## log gradient info once every epoch (avoid clutter)
            self.log_train({'gradient':grad/num_samples})

            self.checkpointing(epoch)

            self.scheduling()

            if converged:
                logging.info('Stopping!! Model has low gradient')
                break
