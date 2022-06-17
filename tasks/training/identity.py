
import torch

from runners import TrainRunner
## conflict with configuration file in runers (cyclic dependency)


class IdentityTrainRunner(TrainRunner):

    def forward_model(self, batch, model, experiment):

        ## sample random points -4 to 4
        points2D_source = 4 * torch.rand([1024, 2], device=self.device) - 2.0

        ## forward
        points3D_target, points2D_target, points3D_source = model(points2D_source)

        model_out = [points3D_target, points2D_target, points2D_source, points3D_source]

        ## try to reproduce input (ie function is identity)
        loss = (points2D_target - points2D_source).pow(2).sum(-1).mean()
        logs = { 'loss': loss.detach() }

        return model_out, loss, logs


    def regularizations(self, model, experiment, predictions, batch, logs):
        loss = 0.0
        return loss, logs
