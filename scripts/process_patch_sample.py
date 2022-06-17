
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from utils import list_all_files

from .patches_function import extract_patches


parser = ArgumentParser(description='convert a mesh into a Neural Convolutional Surfaces sample')
parser.add_argument('--data', type=str, help='path to file or folder containing meshes to convert', required=True)
parser.add_argument('--circle', action='store_true', default=False, help='use circular slim (default is square)')
parser.add_argument('--patch_size', type=float, default=0.1, help='patch size')
parser.add_argument('--rand_seed', type=int, default=42, help='random seed')
parser.add_argument('--global_param', action='store_true', default=False, help='compute global parametrization (for NCS must be on)')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
args = parser.parse_args()


#############
np.random.seed(args.rand_seed)

file_list = list_all_files(args.data, mesh=True, binary=False)

bnd_shape = 'square' if not args.circle else 'circle'

for file in tqdm(file_list):
    name = file.split('/')[-1]
    name = '.'.join(name.split('.')[:-1])

    if args.verbose:
        print(file, name)

    extract_patches(file, name, bnd_shape, args.patch_size, args.global_param, verbose=args.verbose)
