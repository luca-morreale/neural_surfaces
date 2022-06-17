
from runners import TrainRunner


class SurfaceMapTrainRunner(TrainRunner):
    ## surface to surface map trainer

    def forward_model(self, batch, model, experiment):

        points2D_source = batch['source_points']
        R               = batch['R']
        t               = batch['t']
        C_target        = batch['C_target']
        C_source        = batch['C_source']

        target_domain = batch['target_domain']

        points2D_source.requires_grad_(True)

        points3D_target, points2D_target, points3D_source = model(points2D_source, [R,t])
        points3D_target *= C_target
        points3D_source *= C_source

        model_out = [points3D_target, points2D_target, points2D_source, points3D_source]
        loss, logs = experiment['loss'](points3D_target, points2D_target, points2D_source, points3D_source, target_domain)

        return model_out, loss, logs


    def regularizations(self, model, experiment, predictions, batch, logs):
        loss = 0.0

        if experiment['loss'].reg_boundary > 0.0:
            source_boundary = batch['boundary']
            R               = batch['R']
            t               = batch['t']

            source_boundary_mapped = model.forward_map(source_boundary, [R, t])

            loss_boundary, _ = experiment['loss'].boundary_loss(source_boundary_mapped, source_boundary)

            logs['loss_boundary'] = loss_boundary.detach()
            loss += experiment['loss'].reg_boundary * loss_boundary


        if experiment['loss'].reg_landmarks > 0.0:
            target_landmarks = batch['target_landmarks']
            landmarks        = batch['landmarks']
            R                = batch['R']
            t                = batch['t']

            landmarks_mapped = model.forward_map(landmarks, [R, t])

            ## Compute landmark loss in 2D
            # loss_lands = (target_landmarks - landmarks_mapped).pow(2).sum(-1).mean()

            ## Compute landmark loss in 3D
            landmarks3D_mapped = model.target_surface(landmarks_mapped)
            landmarks3D_target = model.target_surface(target_landmarks)
            loss_lands = (landmarks3D_target - landmarks3D_mapped).pow(2).sum(-1).mean()

            loss += experiment['loss'].reg_landmarks * loss_lands
            logs['loss_landmarks'] = loss_lands.detach()

        return loss, logs
