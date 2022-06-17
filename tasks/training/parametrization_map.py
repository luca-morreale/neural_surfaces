
from runners import TrainRunner


class ParametrizationMapTrainRunner(TrainRunner):
    ## surface to surface map trainer

    def forward_model(self, batch, model, experiment):

        points2D_source = batch['source_points']

        points2D_source.requires_grad_(True)

        points2D_target, points3D_source = model(points2D_source)

        model_out = [points2D_target, points2D_source, points3D_source]
        loss, logs = experiment['loss'](points2D_target, points2D_source, points3D_source)

        return model_out, loss, logs


    def regularizations(self, model, experiment, predictions, batch, logs):
        loss = 0.0
        return loss, logs
