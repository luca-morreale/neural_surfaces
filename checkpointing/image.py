
from .mixin import Mixin


class ImageCheckpointing(Mixin):

    def check_imports(self):
        ### check if library exists, if does not remove function
        try:
            from matplotlib import pyplot
        except ImportError as err:
            print('Error missing library ' + str(err))
            self.plot_grad_flow        = self.empty_function
            self.uv       = self.empty_function
            self.save_landmarks_image  = self.empty_function
            self.save_uvmesh_overlap     = self.empty_function
            self.save_uvmesh_colormap    = self.empty_function
            self.save_uvmesh_domain      = self.empty_function
            self.save_grad_directions  = self.empty_function


    # ========================= Visualize gradients ======================= #
    def plot_grad_flow(self, named_parameters, checkpoint_dir, prefix=''):
        from utils import save_grad_barplot
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        med_grads = []
        avg_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                med_grads.append(p.grad.abs().median().cpu())
                avg_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())

        out_folder = self.compose_out_folder(checkpoint_dir, ['gradients'])
        filename = '{}/{}gradients.png'.format(out_folder, prefix)
        save_grad_barplot(filename, max_grads, avg_grads, layers)
        filename = '{}/{}avggradients.png'.format(out_folder, prefix)
        save_grad_barplot(filename, [], avg_grads, layers)
        filename = '{}/{}medgradients.png'.format(out_folder, prefix)
        save_grad_barplot(filename, [], med_grads, layers)


    # ========================= Individual visualizations =========================== #

    def save_uvmesh_image(self, points, faces, checkpoint_dir, prefix=''):
        ### Save mesh layout in 2D
        from utils import save_uv_layout
        out_folder = self.compose_out_folder(checkpoint_dir, ['parametrization'])
        filename = '{}/{}layout.png'.format(out_folder, prefix)
        save_uv_layout(filename, points.squeeze(), faces)

    def save_landmarks_image(self, points, faces, landmarks, checkpoint_dir, prefix=''):
        ### Save mesh layout in 2D w/ landmarks
        from utils import save_uv_layout
        out_folder = self.compose_out_folder(checkpoint_dir, ['parametrization'])
        filename = '{}/{}layout_wlandmarks.png'.format(out_folder, prefix)
        save_uv_layout(filename, points.squeeze(), faces, landmarks)

    def save_uvmesh_overlap(self, grids, faces, checkpoint_dir, prefix=''):
        ### Save multiple mesh layout in 2D with different colors
        from utils import save_overlapping_uv_layout
        out_folder = self.compose_out_folder(checkpoint_dir, ['parametrization'])
        filename = '{}/{}layouts_overlap.png'.format(out_folder, prefix)
        save_overlapping_uv_layout(filename, grids, faces)

    def save_uvmesh_colormap(self, points, scalar, checkpoint_dir, prefix=''):
        ### Save 2D points w/ colors based on scalars
        from utils import save_uv_layout_colormap
        out_folder = self.compose_out_folder(checkpoint_dir, ['parametrization'])
        filename = '{}/{}colormap.png'.format(out_folder, prefix)
        save_uv_layout_colormap(filename, points, scalar)

    def save_uvmesh_domain(self, points, checkpoint_dir, prefix=''):
        from utils import save_domain_image
        out_folder = self.compose_out_folder(checkpoint_dir, ['parametrization'])
        filename = '{}/{}domain.png'.format(out_folder, prefix)
        save_domain_image(filename, points)

    def save_grad_directions(self, uv_points, directions, faces, checkpoint_dir, prefix=''):
        from utils import save_uv_displacement
        out_folder = self.compose_out_folder(checkpoint_dir, ['parametrization'])
        filename = '{}/{}gradient.png'.format(out_folder, prefix)
        save_uv_displacement(filename, uv_points, directions, triangles=faces)
