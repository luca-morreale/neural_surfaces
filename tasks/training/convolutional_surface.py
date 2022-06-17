
from .pca_surface import PCASurfaceTrainer


class ConvSurfaceTrainer(PCASurfaceTrainer):
    ## trainer for NCS

    def forward_model(self, batch, model, experiment):
        coarse_param = batch['coarse_param']
        param        = batch['param']
        gt           = batch['gt']
        idx          = batch['idx']
        _lambda      = batch['lambda'] if 'lambda' in batch else 1.0

        coarse_param.requires_grad_(True)
        param.requires_grad_(True)

        points, points_uvs = model(param, idx, coarse_param, return_uvs=True)

        loss, logs = experiment['loss'](points, gt)
        loss_uvs, logs_uvs = experiment['loss'](points_uvs, gt)

        logs['loss_distance'] = loss.item()
        logs['loss_uv'] = logs_uvs['loss'].item()

        ## weight the loss based on lambda (coarse to fine adjustment)
        loss = (1.0 - _lambda) * loss_uvs + _lambda * loss

        return (points, points_uvs), loss, logs


    def regularizations(self, model, experiment, predictions, batch, logs):
        points     = predictions[0]
        points_uvs = predictions[0]
        loss = 0.0

        if experiment['loss'].reg_normals > 0.0:
            normals = batch['normals'].view(-1, 3)
            mask    = batch['mask'].view(-1)

            ## coarse normals
            pred_normals = self.compute_normals(out=points_uvs, wrt=batch['coarse_param']).view(-1, 3)
            loss_norm_coarse, logs_norm_coarse = experiment['loss'](pred_normals[mask], normals[mask])
            logs['loss_norm_coarse'] = logs_norm_coarse['loss']
            loss += (1.0 - batch['lambda']) * experiment['loss'].reg_normals * loss_norm_coarse

            ## normals
            pred_normals = self.compute_normals(out=points, wrt=batch['coarse_param']).view(-1, 3)
            loss_norm, logs_norm = experiment['loss'](pred_normals[mask], normals[mask])
            logs['loss_norm'] = logs_norm['loss']
            loss += batch['lambda'] * experiment['loss'].reg_normals * loss_norm

            ### fine normals
            #pred_normals = self.compute_normals(out=points, wrt=batch['param']).view(-1, 3)
            #loss_norm, logs_norm = experiment['loss'](pred_normals[mask], normals[mask])
            #logs['loss_norm_fine'] = logs_norm['loss']
            #loss += batch['lambda'] * experiment['loss'].reg_normals * loss_norm

            logs['loss_norm'] = logs_norm['loss'] + logs_norm_coarse['loss']

        return loss, logs
