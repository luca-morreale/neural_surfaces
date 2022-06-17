
from .surface import SurfaceTrainer

class PCASurfaceTrainer(SurfaceTrainer):
    ## trainer for PCA NCS

    def forward_model(self, batch, model, experiment):
        param   = batch['param']
        gt      = batch['gt']
        idx     = batch['idx']
        Cs      = batch['Cs']

        if experiment['loss'].reg_normals > 0.0:
            param.requires_grad_(True)

        points = model(param, idx) / Cs
        loss, logs = experiment['loss'](points, gt)
        logs['loss_distance'] = loss.detach()

        return points, loss, logs
