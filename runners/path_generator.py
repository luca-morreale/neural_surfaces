
import json
import os
from datetime import datetime

from utils import mkdir


class ConfigPathGenerator():

    ### Compose folders name from exp parameters
    def compose_config_path(self):
        exp_str = self.stringify_experiment()

        timestamp = str(datetime.timestamp(datetime.now()))

        log_folder = self.config['logging']['base_path']

        namespace  = self.config['logging']['namespace']
        log_folder += namespace + '_'

        if self.config['logging']['identifier'] is not None and self.config['logging']['identifier'] != '':
            namespace  += '_' + str(self.config['logging']['identifier'])
            log_folder += str(self.config['logging']['identifier']) + '_'

        self.config['logging']['folder'] = log_folder + exp_str

        self.config['checkpointing']['folder'] = self.config['checkpointing']['base_path'] + namespace + '_' + exp_str + '_' + timestamp

        self.copy_config_to_experiment_folder()


    ### Copy configuration to experiment folder
    def copy_config_to_experiment_folder(self):

        folder = self.config['checkpointing']['folder']
        mkdir(folder)

        out_path = os.path.join(folder, 'config.json')
        with open(out_path, 'w') as outfile:
            json.dump(self.config, outfile, sort_keys=True, indent=4)


    ### Convert exp config to string
    def stringify_experiment(self):
        model_str   = self.stringify_model(self.config['models'])
        dataset_str = self.stringify_dataset(self.config['datasets'])
        loss_str    = self.stringify_loss(self.config['loss'])
        opt_str     = self.stringify_optimizer(self.config['optimizers'])
        sch_str     = self.stringify_scheduler(self.config['schedulers'])

        out_str = model_str + '_' + loss_str

        if len(dataset_str) > 1:
            out_str += '_' + dataset_str
        out_str += '_' + opt_str

        if len(sch_str) > 1:
            out_str += '_' + sch_str

        if ('load_checkpoint', True) in self.config['models'].items():
            out_str += '_restart'

        return out_str


    def stringify_dataset(self, config):
        out_str = ''
        if 'batch_size' in config['train'] and ('batch_size', None) not in config['train'].items():
            if config['train']['batch_size'] > 1:
                out_str += 'BS-' + str(config['train']['batch_size'])
        if 'num_points' in config['train']:
            if len(out_str) > 1:
                out_str += '_'
            out_str += 'NP-' + str(config['train']['num_points'])

        return out_str


    def stringify_loss(self, config):
        out_str = 'loss-'  + config['name']


        for reg_name, short in {'reg_lambda':'Wlamda',
                            'reg_incremental':'Wincrem',
                            'reg_sparsity':'Wsparse',
                            'reg_normals':'Wnormals',
                            'reg_ortho':'Wortho',
                            'reg_folding':'Wfold',
                            'reg_distortion':'Wdist',
                            'reg_domain':'Wdomain',
                            'reg_boundary':'Wbound',
                            'reg_param':'Wparam',
                            'reg_landmarks':'Wland',
                            'reg_embeddings':'Wembs',
                            'reg_outside':'Wout',
                        }.items():
            if reg_name in config['params']:
                if config['params'][reg_name] > 0.0:
                    out_str += '_{}-{:.1e}'.format(short, config['params'][reg_name])


        return out_str

    def stringify_optimizer(self, config_opts):
        for config in config_opts:
            out_str =  'opt-' + config['name'] + '_'
            opt_params = config['params']

            out_str += 'LR-{:.1e}'.format(opt_params['lr'])
            if 'momentum' in opt_params:
                if opt_params['momentum'] != 0.0:
                    out_str += '_'
                    if 'nesterov' in opt_params:
                        out_str += 'nest'
                    out_str += 'mom-' + str(opt_params['momentum'])

            if 'l2_weight_decay' in opt_params:
                if opt_params['l2_weight_decay'] > 0.0:
                    out_str += '_l2-'  + str(opt_params['l2_weight_decay'])

            if 'clip_grad_value' in opt_params:
                if opt_params['clip_grad_value'] is not None:
                    out_str += '_clipgrad-{:.1e}'.format(opt_params['clip_grad_value'])


            if ('accumulate_gradient', True) in opt_params:
                out_str += '_accgrad' + str(opt_params['accumulation_steps'])
            return out_str

    def stringify_scheduler(self, config):
        out_str = ''
        for sch in config:
            out_str += f'_sch{sch["opt_idx"]}'
            #out_str += f'_sch{sch["opt_idx"]}-{sch["name"]}'
        if len(out_str) > 0:
            out_str = out_str[1:]
        return out_str



    def stringify_model(self, config):
        out_str = config['name'] + '_'

        if 'ConvSurface' in config['name']:
            out_str += 'H-' + str(config['structure']['cnn']['latent_size']) + '_'
            out_str += 'C-' + str(config['structure']['cnn']['latent_depth']) + '_'

            out_str += 'MLP-' + self.list_to_string(config['structure']['mlp']['layers'], '-') + '_'

            if 'coarse_mlp' in config['structure']:
                out_str += 'MLPcoarse-' + self.list_to_string(config['structure']['coarse_mlp']['layers'], '-') + '_'

            out_str += 'channels-' + self.list_to_string(config['structure']['cnn']['channels'], '-') + '_'
            out_str += 'kernels-' + self.list_to_string(set(config['structure']['cnn']['kernels']), '-') + '_'

            out_str += 'CNN'+self.stringify_layer_info(config['structure']['cnn']) + '_'
            out_str += 'MLP'+self.stringify_layer_info(config['structure']['mlp'])
        elif config['name'] == 'PointInterpolation':
            out_str += 'H-' + str(config['structure']['encoding_size'])
        elif config['name'] == 'MLP' or config['name'] == 'ResidualMLP':
            out_str += self.list_to_string(config['structure']['layers'], '-') + '_'
            out_str += self.stringify_layer_info(config['structure'])
        elif config['name'] == 'NeuralMap' or config['name'] == 'PositionalNeuralMap':
            out_str += self.list_to_string(config['structure']['map']['layers'], '-') + '_'
            out_str += self.stringify_layer_info(config['structure']['map'])
        else:
            out_str += self.list_to_string(config['structure']['layers'], '-') + '_'
            out_str += self.stringify_layer_info(config['structure'])

        return out_str


    def stringify_layer_info(self, config):
        out_str = ''
        if 'norm' in config and ('norm', None) not in config.items():
            out_str += 'wnorm-' + str(config['norm']) + '_'
        if 'drop' in config and ('drop', 0.0) not in config.items():
            out_str += 'wdrop-' + str(config['drop']) + '_'
        if 'act' in config and ('act', None) not in config.items():
            out_str += 'wact-' + config['act'] # + '_'
            # out_str += 'wmlpact-' + (config['structure']['mlpact'] if config['structure']['mlpact'] is not None else 'identity') # + '_'
        else:
            out_str += 'wact-' + (config['act'] if config['act'] is not None else 'identity')
        # out_str += 'winit-' + config.structure.init

        return out_str


    def list_to_string(self, my_list, token):
        return token.join([str(el) for el in my_list])

    def outlist_to_string(self, my_list, token):
        return token.join([str(el[len(el)-1]) for el in my_list])
