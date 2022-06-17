
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from utils import list_all_files


def assemble_correspondences(shape):
    corresp  = [ [] for _ in shape['global']['points'] ]
    for i, patch in enumerate(shape['patches']):
        for j, idx in enumerate(patch['idx']):
            corresp[idx].append([i, j])

    return corresp


def compute_weights(shape, corresp):

    weights = [ 1.0-item['param'].abs().max(dim=1)[0] for item in shape['patches'] ]

    for el in corresp:
        if len(el) <= 1:
            weights[el[0][0]][el[0][1]] = 1.0
        w_sum = 0
        for i, j in el:
            w_sum += weights[i][j]
        if w_sum < 1.0e-6:
            for i, j in el:
                weights[i][j] = 1.0 / len(el)

    return weights



parser = ArgumentParser(description='compute visual interpolation weights')
parser.add_argument('--data', type=str, help='path to input pth sample or folder', required=True)
args = parser.parse_args()


files_list = list_all_files(args.data, binary=True, mesh=False)

for file in tqdm(files_list):

    data = torch.load(file)

    corresp = assemble_correspondences(data)
    weights = compute_weights(data, corresp)

    data['global']['weights'] = weights

    torch.save(data, file)
