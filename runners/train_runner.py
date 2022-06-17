
from .generic_runner import GenericRunner

class TrainRunner(GenericRunner):

    ## base function for training
    def run(self, batch, model, experiment):

        ## call model and compute main losses
        model_out, loss, logs = self.forward_model(batch, model, experiment)
        ## compute regularization terms if any
        loss_reg, logs = self.regularizations(model, experiment, model_out, batch, logs)
        ## add regularization to loss
        loss += loss_reg
        ## update log of loss
        logs['loss'] = loss.detach()

        return loss, logs
