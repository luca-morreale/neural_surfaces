
import torch

from .mixin import Mixin


class BinaryCheckpointing(Mixin):

    # ================================================================== #
    # =================== Save model =================================== #
    def save_model(self, checkpoint_folder, model, name=''):

        folder = self.compose_out_folder(checkpoint_folder, ['models'])
        file_name = 'model{}'.format(name)

       # save last model
        model_path = '{}/{}.pth'.format(folder, file_name)
        torch.save(model.state_dict(), model_path)
    # ================================================================== #