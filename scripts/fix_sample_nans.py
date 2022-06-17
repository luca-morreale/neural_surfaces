
import numpy as np
import torch
from argparse import ArgumentParser

from utils import parametrize
from utils import list_all_files

def fix_param_nans(filename, bnd_shape):
    data = torch.load(filename)

    if 'patches' not in data: # not a collection sample
        return

    for key in data['global'].keys():
        if key == 'name':
            continue
        if type(data['global'][key]) != torch.Tensor:
            if np.isnan(data['global'][key]):
                print(data['global'][key])
            continue
        if torch.isnan(data['global'][key]).any():
            print(f'Found NaNs global {key} {filename}')
            P = np.array(data['global']['points'].tolist())
            uv_slim_global = parametrize(P, data['global']['faces'].numpy(), 'slim', it=100, bnd_shape=bnd_shape)
            data['global']['param'] = torch.from_numpy(uv_slim_global).float()

    for i, patch in enumerate(data['patches']):
        for key in patch.keys():
            if type(patch[key]) != torch.Tensor:
                if np.isnan(patch[key]):
                    print(patch[key])
                continue
            if torch.isnan(patch[key]).any():
                print(f'Found NaNs {key} {filename}')

        if torch.isnan(patch['param']).sum() > 0:
            points = np.array(patch['points'].tolist())
            uv_slim_global = parametrize(points, patch['faces'].numpy(), 'slim', it=100, bnd_shape=bnd_shape)
            data['patches'][i]['param'] = torch.from_numpy(uv_slim_global).float()

            print(f'Replacing parametrization {i} {filename}')
            if torch.isnan(patch['param']).any():
                print('Failed!')
            else:
                print('Succeded!')

    torch.save(data, filename)



parser = ArgumentParser(description='fix nans in parametrization of a sample if there is any')
parser.add_argument('--data', type=str, help='path to file or folder containing pth samples to check', required=True)
parser.add_argument('--square', action='store_true', default=False, help='use squared slim (default circle)')
parser.add_argument('--free', action='store_true', default=False, help='use free slim (default circle)')
args = parser.parse_args()

bnd_shape = 'square' if args.square else ('free' if args.free else 'circle')

files_list = list_all_files(args.data, binary=True, mesh=False)

for filename in files_list:
    fix_param_nans(filename, bnd_shape)
